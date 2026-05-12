from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import uuid
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
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
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
    "Wall time spent in the pyannote inference call (excluding upload and queue wait)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0),
)
AUDIO_DURATION_SECONDS = Histogram(
    "pyannote_audio_duration_seconds",
    "Input audio duration in seconds",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0),
)
REALTIME_FACTOR = Histogram(
    "pyannote_realtime_factor",
    "Inference wall time divided by audio duration (lower is faster; <1 = faster than realtime)",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)
SSE_RESULTS_TOTAL = Counter(
    "pyannote_sse_results_total",
    "Terminal SSE events emitted by /diarize",
    labelnames=("outcome",),
)
QUEUE_DEPTH = Gauge(
    "pyannote_queue_depth",
    "Jobs currently waiting in the diarization queue (excludes jobs being processed)",
)
ACTIVE_REQUESTS = Gauge("pyannote_active_requests", "Jobs currently being processed by a worker")
MODEL_LOADED = Gauge("pyannote_model_loaded", "Whether the diarization pipeline is loaded (1=yes)")
PYANNOTE_BUILD = Info(
    "pyannote",
    "Static build/runtime information for the diarization service",
)

SSE_RESULTS_TOTAL.labels(outcome="success")
SSE_RESULTS_TOTAL.labels(outcome="error")

_BEARER_SCHEME = HTTPBearer(bearerFormat="API key", auto_error=False)


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

DIARIZE_WORKERS = max(1, int(os.environ.get("DIARIZE_WORKERS", "1").strip() or "1"))
SSE_HEARTBEAT_SECONDS = max(
    0.5, float(os.environ.get("SSE_HEARTBEAT_SECONDS", "5").strip() or "5")
)
MAX_QUEUE_DEPTH = max(1, int(os.environ.get("MAX_QUEUE_DEPTH", "64").strip() or "64"))


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


class _DiarizationParams(BaseModel):
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    exclusive: bool = False


class _Job:
    """One queued diarization request and its private event stream."""

    def __init__(self, tmp_path: Path, params: _DiarizationParams) -> None:
        self.id: str = uuid.uuid4().hex
        self.tmp_path: Path = tmp_path
        self.params: _DiarizationParams = params
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()


_JOB_QUEUE: asyncio.Queue[_Job] | None = None
_WORKER_TASKS: list[asyncio.Task[None]] = []


def _run_diarization_sync(tmp_path: Path, params: _DiarizationParams) -> DiarizeResponse:
    """Blocking diarization pipeline call; intended for asyncio.to_thread."""
    t0 = time.monotonic()
    waveform, sample_rate = torchaudio.load(str(tmp_path))
    if waveform.dim() != 2:
        raise HTTPException(status_code=400, detail={"error": "invalid_audio"})
    audio_duration = float(waveform.shape[1]) / float(sample_rate)
    AUDIO_DURATION_SECONDS.observe(audio_duration)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    waveform = waveform.to(device)

    infer_kwargs: dict[str, Any] = {}
    if params.num_speakers is not None:
        infer_kwargs["num_speakers"] = params.num_speakers
    if params.min_speakers is not None:
        infer_kwargs["min_speakers"] = params.min_speakers
    if params.max_speakers is not None:
        infer_kwargs["max_speakers"] = params.max_speakers

    t_infer0 = time.monotonic()
    raw_out = _pipeline({"waveform": waveform, "sample_rate": sample_rate}, **infer_kwargs)
    inference_seconds = time.monotonic() - t_infer0
    DIARIZATION_SECONDS.observe(inference_seconds)
    if audio_duration > 0:
        REALTIME_FACTOR.observe(inference_seconds / audio_duration)

    try:
        diarization = _extract_diarization_output(raw_out, exclusive=params.exclusive)
    except (KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "diarization_output_parse_failed"},
        ) from exc

    segments, speakers = _annotation_to_segments(diarization)
    return DiarizeResponse(
        duration_seconds=audio_duration,
        num_speakers=len(speakers),
        speakers=speakers,
        segments=segments,
        processing_time_seconds=time.monotonic() - t0,
        pyannote_version=_pyannote_version,
    )


async def _worker(worker_id: int) -> None:
    """Pull jobs off the shared queue and run the pyannote pipeline in a thread."""
    assert _JOB_QUEUE is not None
    while True:
        job = await _JOB_QUEUE.get()
        QUEUE_DEPTH.dec()
        ACTIVE_REQUESTS.inc()
        try:
            await job.events.put({"type": "status", "phase": "running", "worker": worker_id})
            logger.info("job_started job=%s worker=%s", job.id, worker_id)
            try:
                result = await asyncio.to_thread(_run_diarization_sync, job.tmp_path, job.params)
            except HTTPException as exc:
                await job.events.put(
                    {"type": "error", "status": exc.status_code, "detail": exc.detail}
                )
            except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
                logger.exception("diarization_failed job=%s err=%s", job.id, exc)
                await job.events.put(
                    {"type": "error", "status": 500, "detail": {"error": "diarization_failed"}}
                )
            else:
                await job.events.put({"type": "result", "data": result.model_dump()})
                logger.info("job_finished job=%s worker=%s", job.id, worker_id)
        finally:
            ACTIVE_REQUESTS.dec()
            try:
                job.tmp_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not delete temp upload %s: %s", job.tmp_path, exc)
            _JOB_QUEUE.task_done()


def _sse_frame(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _pipeline, _pyannote_version, _JOB_QUEUE
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
    _JOB_QUEUE = asyncio.Queue(maxsize=MAX_QUEUE_DEPTH)
    for i in range(DIARIZE_WORKERS):
        _WORKER_TASKS.append(asyncio.create_task(_worker(i), name=f"diarize-worker-{i}"))
    app.state.ready = True
    MODEL_LOADED.set(1)
    logger.info(
        'Attribution: "%s" by pyannote / pyannoteAI; license CC-BY-4.0 (%s); model card %s.',
        MODEL_ID,
        MODEL_LICENSE_URL,
        MODEL_CARD_URL,
    )
    logger.info(
        "Pipeline loaded; ready to serve. workers=%d max_queue_depth=%d heartbeat=%.1fs",
        DIARIZE_WORKERS,
        MAX_QUEUE_DEPTH,
        SSE_HEARTBEAT_SECONDS,
    )
    yield
    MODEL_LOADED.set(0)
    app.state.ready = False
    for task in _WORKER_TASKS:
        task.cancel()
    await asyncio.gather(*_WORKER_TASKS, return_exceptions=True)
    _WORKER_TASKS.clear()
    _JOB_QUEUE = None
    _pipeline = None


app = FastAPI(title="pyannote speaker diarization (community-1)", lifespan=lifespan)


def _require_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER_SCHEME)],
) -> None:
    if credentials is None or credentials.credentials not in ALLOWED_API_KEYS:
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


@app.post("/diarize")
async def diarize(
    _auth: Annotated[None, Depends(_require_api_key)],
    request: Request,
    file: Annotated[UploadFile, File(..., description="Audio file")],
    num_speakers: Annotated[int | None, Query()] = None,
    min_speakers: Annotated[int | None, Query()] = None,
    max_speakers: Annotated[int | None, Query()] = None,
    exclusive: Annotated[bool, Query()] = False,
) -> StreamingResponse:
    if _pipeline is None or _JOB_QUEUE is None:
        raise HTTPException(status_code=503, detail={"error": "pipeline_not_loaded"})

    if _JOB_QUEUE.full():
        logger.warning(
            "queue_full reject endpoint=/diarize depth=%d max=%d",
            _JOB_QUEUE.qsize(),
            MAX_QUEUE_DEPTH,
        )
        raise HTTPException(
            status_code=503,
            detail={"error": "queue_full", "max_queue_depth": MAX_QUEUE_DEPTH},
            headers={"Retry-After": "5"},
        )

    suffix = Path(file.filename or "audio").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed manually below
        prefix="pyannote_upload_", suffix=suffix, delete=False
    )
    tmp_path = Path(tmp.name)
    try:
        chunk_size = 1024 * 1024
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            tmp.write(chunk)
    finally:
        tmp.close()

    params = _DiarizationParams(
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        exclusive=exclusive,
    )
    job = _Job(tmp_path=tmp_path, params=params)
    try:
        _JOB_QUEUE.put_nowait(job)
        QUEUE_DEPTH.inc()
    except asyncio.QueueFull as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as unlink_exc:
            logger.warning("Could not delete temp upload %s: %s", tmp_path, unlink_exc)
        logger.warning(
            "queue_full reject_after_upload depth=%d max=%d",
            _JOB_QUEUE.qsize(),
            MAX_QUEUE_DEPTH,
        )
        raise HTTPException(
            status_code=503,
            detail={"error": "queue_full", "max_queue_depth": MAX_QUEUE_DEPTH},
            headers={"Retry-After": "5"},
        ) from exc
    logger.info(
        "job_queued job=%s queue_depth=%d max=%d",
        job.id,
        _JOB_QUEUE.qsize(),
        MAX_QUEUE_DEPTH,
    )

    async def event_stream() -> AsyncIterator[bytes]:
        t0 = time.monotonic()
        phase = "queued"
        yield _sse_frame("status", {"job_id": job.id, "phase": phase})
        while True:
            if await request.is_disconnected():
                logger.info("client_disconnected job=%s phase=%s", job.id, phase)
                return
            try:
                event = await asyncio.wait_for(
                    job.events.get(), timeout=SSE_HEARTBEAT_SECONDS
                )
            except asyncio.TimeoutError:
                yield _sse_frame(
                    "heartbeat",
                    {
                        "job_id": job.id,
                        "phase": phase,
                        "elapsed_seconds": round(time.monotonic() - t0, 2),
                    },
                )
                continue

            kind = event["type"]
            if kind == "status":
                phase = event["phase"]
                yield _sse_frame(
                    "status",
                    {"job_id": job.id, "phase": phase, "worker": event.get("worker")},
                )
            elif kind == "result":
                SSE_RESULTS_TOTAL.labels(outcome="success").inc()
                yield _sse_frame("result", {"job_id": job.id, **event["data"]})
                return
            elif kind == "error":
                SSE_RESULTS_TOTAL.labels(outcome="error").inc()
                yield _sse_frame(
                    "error",
                    {"job_id": job.id, "status": event["status"], "detail": event["detail"]},
                )
                return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
