from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import torch
import torchaudio
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pydantic import BaseModel, Field

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("pyannote_service")


REQUESTS_TOTAL = Counter(
    "pyannote_requests_total",
    "HTTP requests processed",
    labelnames=("endpoint", "status"),
)
DIARIZATION_SECONDS = Histogram(
    "pyannote_diarization_duration_seconds",
    "Wall time spent in diarization (excluding upload I/O)",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, float("inf")),
)
AUDIO_DURATION_SECONDS = Histogram(
    "pyannote_audio_duration_seconds",
    "Input audio duration in seconds",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 2400.0, float("inf")),
)
ACTIVE_REQUESTS = Gauge("pyannote_active_requests", "Requests currently in-flight")
MODEL_LOADED = Gauge("pyannote_model_loaded", "Whether the diarization pipeline is loaded (1=yes)")
PYANNOTE_BUILD = Info(
    "pyannote",
    "Static build/runtime information for the diarization service",
)

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _read_allowed_api_keys() -> frozenset[str]:
    raw = os.environ.get("API_KEYS")
    if raw is None or not raw.strip():
        logger.error(
            "Refusing to start: API_KEYS is unset or empty; set a comma-separated list of keys."
        )
        sys.exit(1)
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


ALLOWED_API_KEYS: frozenset[str] = _read_allowed_api_keys()

MODEL_PATH_RAW = os.environ.get("MODEL_PATH", "").strip()
MODEL_ID = os.environ.get(
    "MODEL_ID",
    "pyannote/speaker-diarization-community-1",
).strip()

MODEL_CARD_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1"
MODEL_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


class SegmentModel(BaseModel):
    start: float
    end: float
    speaker: str


class DiarizeResponse(BaseModel):
    duration_seconds: float = Field(..., description="Audio duration in seconds")
    num_speakers: int = Field(..., ge=0)
    speakers: list[str]
    segments: list[SegmentModel]
    processing_time_seconds: float
    model: str = "pyannote/speaker-diarization-community-1"
    pyannote_version: str


_pipeline: Any = None
_pyannote_version: str = "unknown"


def _load_pyannote_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("pyannote.audio")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _is_testing() -> bool:
    return os.environ.get("PYANNOTE_TESTING", "").strip() in {"1", "true", "yes"}


def _read_hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val is not None and val.strip():
            return val.strip()
    return None


def _local_pipeline_dir() -> Path | None:
    if not MODEL_PATH_RAW:
        return None
    path = Path(MODEL_PATH_RAW).expanduser()
    if not path.is_dir():
        logger.error(
            "MODEL_PATH=%s must be an existing directory containing a full offline "
            "pyannote pipeline checkout (see %s, Offline use).",
            MODEL_PATH_RAW,
            MODEL_CARD_URL,
        )
        sys.exit(1)
    if not (path / "config.yaml").exists():
        logger.error(
            "MODEL_PATH=%s must include config.yaml from the upstream pipeline snapshot.",
            MODEL_PATH_RAW,
        )
        sys.exit(1)
    return path


class _DryRunPipeline:
    """Minimal stand-in used only when PYANNOTE_TESTING=1 (see tests)."""

    def __call__(self, *_args: Any, **_kwargs: Any) -> dict[str, Annotation]:
        ann = Annotation()
        ann[Segment(0.0, 0.5)] = "SPEAKER_00"
        return {"speaker_diarization": ann}


def _load_pipeline() -> Any:
    if _is_testing():
        logger.warning("PYANNOTE_TESTING is enabled; using dry-run pipeline.")
        return _DryRunPipeline()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    local_dir = _local_pipeline_dir()
    if local_dir is not None:
        logger.info(
            "loading_pipeline source=local path=%s device=%s",
            local_dir,
            device,
        )
        pipe: Pipeline = Pipeline.from_pretrained(str(local_dir))
        pipe.to(device)
        return pipe

    token = _read_hf_token()
    if token is None:
        logger.error(
            "Refusing to start: model weights are not bundled. Either mount an offline "
            "checkout at MODEL_PATH (after you personally accepted HF access terms and "
            "cloned per %s), or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN for your runtime.",
            MODEL_CARD_URL,
        )
        sys.exit(1)
    logger.info(
        "loading_pipeline source=huggingface_hub model_id=%s device=%s",
        MODEL_ID,
        device,
    )
    pipe = Pipeline.from_pretrained(MODEL_ID, token=token)
    pipe.to(device)
    return pipe


def _extract_diarization_output(raw: Any, exclusive: bool) -> Annotation:
    if isinstance(raw, Annotation):
        return raw
    key = "exclusive_speaker_diarization" if exclusive else "speaker_diarization"
    if isinstance(raw, dict):
        if key not in raw:
            raise KeyError(key)
        value = raw[key]
        if not isinstance(value, Annotation):
            raise TypeError(f"{key} is not an Annotation")
        return value
    if hasattr(raw, key):
        value = getattr(raw, key)
        if not isinstance(value, Annotation):
            raise TypeError(f"{key} is not an Annotation")
        return value
    raise TypeError("Unsupported diarization output type")


def _annotation_to_segments(diarization: Annotation) -> tuple[list[SegmentModel], list[str]]:
    segments: list[SegmentModel] = []
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            SegmentModel(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
        )
    speakers_sorted = sorted({s.speaker for s in segments})
    return segments, speakers_sorted


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _pipeline, _pyannote_version
    app.state.ready = False
    MODEL_LOADED.set(0)
    _pyannote_version = _load_pyannote_version()
    logger.info(
        "startup pyannote_version=%s torch_cuda_available=%s model_path=%s model_id=%s hf_home=%s",
        _pyannote_version,
        torch.cuda.is_available(),
        MODEL_PATH_RAW or "(none)",
        MODEL_ID,
        os.environ.get("HF_HOME", ""),
    )
    PYANNOTE_BUILD.info(
        {
            "version": _pyannote_version,
            "model_id": MODEL_ID,
            "torch_version": torch.__version__,
            "cuda_available": str(torch.cuda.is_available()).lower(),
        }
    )
    _pipeline = _load_pipeline()
    app.state.ready = True
    MODEL_LOADED.set(1)
    logger.info(
        'Attribution: "%s" by pyannote / pyannoteAI; license CC-BY-4.0 (%s); model card %s.',
        MODEL_ID,
        MODEL_LICENSE_URL,
        MODEL_CARD_URL,
    )
    logger.info("Pipeline loaded; ready to serve.")
    yield
    MODEL_LOADED.set(0)
    app.state.ready = False
    _pipeline = None


app = FastAPI(title="pyannote speaker diarization (community-1)", lifespan=lifespan)


def _require_api_key(x_api_key: Annotated[str | None, Depends(_API_KEY_HEADER)]) -> None:
    if x_api_key is None or x_api_key not in ALLOWED_API_KEYS:
        raise HTTPException(status_code=401, detail={"error": "unauthorized"})


@app.middleware("http")
async def count_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    response = await call_next(request)
    status = str(response.status_code)
    REQUESTS_TOTAL.labels(endpoint=request.url.path, status=status).inc()
    return response


@app.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    ready = bool(getattr(request.app.state, "ready", False))
    if not ready:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return JSONResponse(status_code=200, content={"status": "ready"})


@app.get("/metrics")
async def metrics() -> Response:
    payload = generate_latest(REGISTRY)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


@app.post("/diarize", response_model=DiarizeResponse)
async def diarize(
    _auth: Annotated[None, Depends(_require_api_key)],
    file: Annotated[UploadFile, File(..., description="Audio file")],
    num_speakers: Annotated[int | None, Query()] = None,
    min_speakers: Annotated[int | None, Query()] = None,
    max_speakers: Annotated[int | None, Query()] = None,
    exclusive: Annotated[bool, Query()] = False,
) -> DiarizeResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail={"error": "pipeline_not_loaded"})
    ACTIVE_REQUESTS.inc()
    tmp_path: Path | None = None
    t0 = time.monotonic()
    try:
        suffix = Path(file.filename or "audio").suffix or ".wav"
        with tempfile.NamedTemporaryFile(
            prefix="pyannote_upload_", suffix=suffix, delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            chunk_size = 1024 * 1024
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                tmp.write(chunk)

        waveform, sample_rate = torchaudio.load(str(tmp_path))
        if waveform.dim() != 2:
            raise HTTPException(status_code=400, detail={"error": "invalid_audio"})
        audio_duration = float(waveform.shape[1]) / float(sample_rate)
        AUDIO_DURATION_SECONDS.observe(audio_duration)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        waveform = waveform.to(device)

        infer_kwargs: dict[str, Any] = {}
        if num_speakers is not None:
            infer_kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            infer_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            infer_kwargs["max_speakers"] = max_speakers

        t_infer0 = time.monotonic()
        raw_out = _pipeline({"waveform": waveform, "sample_rate": sample_rate}, **infer_kwargs)
        DIARIZATION_SECONDS.observe(time.monotonic() - t_infer0)

        try:
            diarization = _extract_diarization_output(raw_out, exclusive=exclusive)
        except (KeyError, TypeError) as exc:
            raise HTTPException(
                status_code=500,
                detail={"error": "diarization_output_parse_failed"},
            ) from exc

        segments, speakers = _annotation_to_segments(diarization)
        processing_time = time.monotonic() - t0
        return DiarizeResponse(
            duration_seconds=audio_duration,
            num_speakers=len(speakers),
            speakers=speakers,
            segments=segments,
            processing_time_seconds=processing_time,
            pyannote_version=_pyannote_version,
        )
    finally:
        ACTIVE_REQUESTS.dec()
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not delete temp upload %s: %s", tmp_path, exc)
