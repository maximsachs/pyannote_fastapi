# pyannote speaker diarization FastAPI Docker image

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Workflow: build-on-push](https://img.shields.io/github/actions/workflow/status/maximsachs/pyannote_fastapi/build-on-push.yml?branch=main&label=build)](https://github.com/maximsachs/pyannote_fastapi/actions/workflows/build-on-push.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/maximsachs/pyannote_fastapi.svg)](https://hub.docker.com/repository/docker/maximfilms/pyannote_fastapi)

<!-- pyannote-version:start -->
**Latest published images build against `pyannote.audio` (not yet released by automation).**
<!-- pyannote-version:end -->

A minimal FastAPI service around [**pyannote/speaker-diarization-community-1**](https://huggingface.co/pyannote/speaker-diarization-community-1). The pipeline is loaded once at startup; diarization requests are queued to in-process workers and streamed back to the client as Server-Sent Events with periodic heartbeats and a final result frame.

**Images:** `docker.io/maximsachs/pyannote_fastapi` and `ghcr.io/maximsachs/pyannote_fastapi` (CUDA `:latest`, CPU `:latest-cpu`).

**Integrating a client?** See [`docs/API.md`](docs/API.md) for the full endpoint reference, every error code, the SSE event schema, and the performance-tuning knobs.

## Model access (weights are not bundled)

The upstream pipeline is **CC-BY-4.0** and **gated** on Hugging Face. This image ships application code only. At runtime you must either:

1. Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) from an account that has accepted the [model card](https://huggingface.co/pyannote/speaker-diarization-community-1) terms, **or**
2. Mount an offline checkout and set `MODEL_PATH` to its directory (must contain `config.yaml`).

## Environment variables

| Name | Required | Default | Notes |
| --- | --- | --- | --- |
| `API_KEYS` | yes | — | Comma-separated accepted keys; clients send one via `Authorization: Bearer <key>`. |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | yes\* | — | \*Unless `MODEL_PATH` is set. |
| `MODEL_PATH` | no | — | Absolute path to a local pipeline checkout. |
| `MODEL_ID` | no | `pyannote/speaker-diarization-community-1` | Hub repo id when `MODEL_PATH` is unset. |
| `HF_HOME` | no | `/opt/huggingface` | Hugging Face cache root (mount a volume here to persist). |
| `DIARIZE_WORKERS` | no | `1` | Concurrent diarization workers. Keep at `1` for single-GPU setups. |
| `MAX_QUEUE_DEPTH` | no | `64` | Max number of jobs that may be queued. Further requests are rejected with `503 {"error":"queue_full"}` and a `Retry-After: 5` header. |
| `SSE_HEARTBEAT_SECONDS` | no | `5` | Interval between SSE `heartbeat` frames while a job is queued or running. |
| `LOG_LEVEL` | no | `INFO` | Python logging level. |
| `PYANNOTE_TELEMETRY` | no | `0` | Set to `1`/`true`/`yes` to opt in to upstream pyannote.audio anonymous usage telemetry. Disabled by default. |

## Quick start

```bash
docker run --rm -it --gpus all \
  -e API_KEYS="replace-me" \
  -e HF_TOKEN="replace-me" \
  -v pyannote_hf_cache:/opt/huggingface \
  -p 8000:8000 \
  ghcr.io/maximsachs/pyannote_fastapi:latest
```

Submit a file and tail the SSE stream:

```bash
curl -N -fsS \
  -H "Authorization: Bearer replace-me" \
  -H "Accept: text/event-stream" \
  -F "file=@/path/to/audio.wav" \
  http://127.0.0.1:8000/diarize
```

`-N` disables curl's output buffering so you see each event as it arrives.

## HTTP API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/live` | Liveness. `200` while the process is up. |
| `GET` | `/health` | Readiness. `200 {"status":"ready"}` once the pipeline is loaded; `503 {"status":"not_ready"}` otherwise. |
| `GET` | `/metrics` | Prometheus exposition. |
| `POST` | `/diarize` | Submit audio, receive an SSE stream of `status` / `heartbeat` events ending in a `result` (or `error`) event. |

**See [`docs/API.md`](docs/API.md)** for the complete request/response schema, every error code, the full SSE event reference, and a client implementation checklist.

**Proxy note:** the SSE response sets `Cache-Control: no-cache` and `X-Accel-Buffering: no`. If you front this with nginx, also set `proxy_buffering off` on the `/diarize` location and make sure idle timeouts on every hop are larger than `SSE_HEARTBEAT_SECONDS`.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
export API_KEYS=dev-key
export PYANNOTE_TESTING=1
uvicorn main:app --app-dir app --reload --host 0.0.0.0 --port 8000
```

```bash
ruff check app tests
pytest -q
```

## License and attribution

Wrapper code is **MIT** ([`LICENSE`](LICENSE)). The model is **CC-BY-4.0**; the service logs an attribution line (model id, license URL, model card URL) at startup. Cached weights on the mounted volume remain gated material — treat the volume as sensitive.
