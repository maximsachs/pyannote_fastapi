# pyannote speaker diarization FastAPI Docker image

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Workflow: build-on-push](https://img.shields.io/github/actions/workflow/status/OWNER/REPO/build-on-push.yml?branch=main&label=build)](https://github.com/OWNER/REPO/actions/workflows/build-on-push.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/DOCKERHUB_USER/IMAGE_NAME.svg)](https://hub.docker.com/r/DOCKERHUB_USER/IMAGE_NAME)

This repository ships a **minimal FastAPI** service around **[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)**: it loads the pipeline once at startup, exposes a small HTTP API with Prometheus metrics, and publishes **CUDA** and **CPU** images that contain **application code only** — **not** redistributed model weights.

## License, attribution, and why weights are not in the image

The upstream pipeline is published on Hugging Face as [**pyannote/speaker-diarization-community-1**](https://huggingface.co/pyannote/speaker-diarization-community-1) under **Creative Commons [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)** (see the model card). **CC-BY-4.0** generally requires **appropriate credit**, a **link to the license**, and **indication of changes** when you share adapted material. This wrapper code is **MIT** ([`LICENSE`](LICENSE)); it does **not** replace or narrow the obligations that apply to **the model weights themselves** when you copy or share them.

The same model card explains that access to files is **gated**: you must **log in and accept the access conditions** on Hugging Face before downloading. **Publishing a public container image with the weights pre-copied would let anyone pull the model without going through that flow**, which is a serious compliance risk against **[Hugging Face’s terms](https://huggingface.co/terms)** and the intent of model-gating — even if the weights are technically CC-BY-4.0.

**This project therefore does not bake weights into the image.** At runtime you must either:

1. Provide your own **`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`** (from an account that has accepted the model conditions) so `pyannote` can download/cache artifacts, **or**
2. Mount an **offline checkout** you obtained lawfully (for example after accepting the conditions and running `git lfs clone` per the model card’s **Offline use** section) and point **`MODEL_PATH`** at that directory.

The service logs a one-line **attribution** string at startup (model id, CC-BY-4.0 link, and model card URL). *This is not legal advice; confirm redistribution with counsel if you ship snapshots or derivative images.*

## Secrets and CI configuration

Configure the following **GitHub Actions secrets** for publishing images:

| Secret | Purpose |
| --- | --- |
| `DOCKERHUB_USERNAME` | Docker Hub namespace used for publishing. |
| `DOCKERHUB_TOKEN` | Docker Hub access token with push permission. |

`GITHUB_TOKEN` is provided automatically for GHCR pushes.

**CI builds do not need `HF_TOKEN`**: images contain no gated weights.

**Runtime / deployment** (your cluster or `docker run`):

| Secret / env | Purpose |
| --- | --- |
| `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` | Hugging Face token with access to community-1, used **only inside your running container** to download/cache the pipeline via `huggingface_hub`. |
| `API_KEYS` | Comma-separated API keys for this HTTP service (same as before). |

**Minimal runtime (typical hub setup):** set only **`API_KEYS`** and **`HF_TOKEN`** (or `HUGGING_FACE_HUB_TOKEN`). The image already sets **`HF_HOME=/opt/huggingface`**; Hugging Face uses that as the cache root. Mount a volume at **`/opt/huggingface`** if you want downloads to survive restarts — **no extra cache-related env** is required.

If you mount the cache somewhere else, set **`HF_HOME`** to the **same path** as `volumeMounts.mountPath` (standard Hugging Face convention).

## Environment variables (runtime)

| Name | Required | Default | Notes |
| --- | --- | --- | --- |
| `API_KEYS` | **Yes** | _none_ | Comma-separated accepted API keys (rotation = multiple active keys; each request sends one `X-API-Key` matching **any** entry). |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | **Yes\*** | _none_ | \*Required **unless** `MODEL_PATH` points at a full offline pipeline directory (must contain `config.yaml`). |
| `MODEL_PATH` | No | _(unset)_ | Absolute path to your **local** checkout of the pipeline (bind-mount or `emptyDir` populated by an init container). When unset, the hub id `MODEL_ID` is used. |
| `MODEL_ID` | No | `pyannote/speaker-diarization-community-1` | Hub repo id used when `MODEL_PATH` is unset. |
| `HF_HOME` | No | `/opt/huggingface` (**set in image**) | Hugging Face cache directory (`hub/`, etc.). Override **only** if you mount storage at a non-default path — **must match** the container mount path. |
| `LOG_LEVEL` | No | `INFO` | Python logging level for the service. |

## Quick start

After accepting the access conditions on the [model card](https://huggingface.co/pyannote/speaker-diarization-community-1) and creating a token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens):

```bash
docker run --rm -it --gpus all \
  -e API_KEYS="replace-me" \
  -e HF_TOKEN="replace-me" \
  -v pyannote_hf_cache:/opt/huggingface \
  -p 8000:8000 \
  ghcr.io/OWNER/REPO:latest
```

```bash
curl -fsS http://127.0.0.1:8000/live
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/metrics | head

curl -fsS \
  -H "X-API-Key: replace-me" \
  -F "file=@/path/to/audio.wav" \
  http://127.0.0.1:8000/diarize
```

Files under `examples/` mirror these flows (`docker-run.sh`, `curl-example.sh`, `k8s-deployment.yaml`).

## HTTP API

### `GET /live`

Liveness: returns `200` if the process is running.

### `GET /health`

Readiness: returns `200` with `{"status":"ready"}` once the diarization pipeline has finished loading; `503` with `{"status":"not_ready"}` beforehand.

### `GET /metrics`

Prometheus exposition format (includes standard process collectors where supported by `prometheus-client`).

### `POST /diarize`

- **Auth:** `X-API-Key` header must match one of the configured `API_KEYS`.
- **Body:** `multipart/form-data` with field name `file`.
- **Optional query parameters:** `num_speakers`, `min_speakers`, `max_speakers`, `exclusive` (boolean; selects `exclusive_speaker_diarization` when supported).

Example JSON response:

```json
{
  "duration_seconds": 123.45,
  "num_speakers": 2,
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "segments": [{"start": 0.21, "end": 3.84, "speaker": "SPEAKER_00"}],
  "processing_time_seconds": 4.2,
  "model": "pyannote/speaker-diarization-community-1",
  "pyannote_version": "3.3.1"
}
```

**Operational note:** requests are processed synchronously — configure your reverse proxy / ingress timeouts accordingly. First startup with hub download can take several minutes; probe timings and `HEALTHCHECK` `start-period` are set generously in the Dockerfiles.

## Image tagging scheme

| Tag pattern | Meaning |
| --- | --- |
| `latest`, `latest-cpu` | Most recent **`main`** build defaults (see workflow env `PYANNOTE_VERSION`). |
| `sha-<short>` | Immutable commit build on `main`. |
| `<semver>`, `<major>.<minor>`, `<major>` | Published by the scheduled PyPI-driven workflow for a released `pyannote.audio` version. |
| `<semver>-cpu`, … | CPU-only variant (smaller image; no GPU required). |

## How automatic builds work

1. **`build-on-push.yml`** builds and pushes on every `main` push using the workflow’s `PYANNOTE_VERSION` pin (keep this aligned with what you want `latest` to track).
2. **`release-on-new-pyannote.yml`** runs **daily at 04:00 UTC** (and on `workflow_dispatch`): it reads the **latest `pyannote.audio` from PyPI**, skips work if Docker Hub already has that semver tag, otherwise builds/pushes the semver-tagged matrix and opens a GitHub Release.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
export API_KEYS=dev-key
export PYANNOTE_TESTING=1
uvicorn main:app --app-dir app --reload --host 0.0.0.0 --port 8000
```

Run checks:

```bash
ruff check app tests
pytest -q
```

## Security notes

- Public images **do not embed** `HF_TOKEN`, `API_KEYS`, or community-1 weights.
- Mount a volume at **`/opt/huggingface`** (the image’s default **`HF_HOME`**) so Hub snapshots persist — **or** mount elsewhere and set **`HF_HOME`** to that path. Cached blobs are still gated material: **treat the volume as sensitive**.
- Keep production secrets in your orchestrator (`Kubernetes` `Secret`, Docker `--env-file` with a **gitignored** `.env`, etc.). `.gitignore` ignores local `.env*` while keeping sample manifests in `examples/`.

## Building images locally

No Hugging Face token is required **to build** — only to **run** the container (unless you mount an offline `MODEL_PATH`):

```bash
docker build \
  --build-arg PYANNOTE_VERSION=3.3.1 \
  -t pyannote-diarization:local \
  .
```

CPU variant:

```bash
docker build \
  --build-arg PYANNOTE_VERSION=3.3.1 \
  -f Dockerfile.cpu \
  -t pyannote-diarization:local-cpu \
  .
```
