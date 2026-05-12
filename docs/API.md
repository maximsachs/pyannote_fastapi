# API reference and integration guide

This document is the source of truth for clients integrating with the diarization service: every endpoint, every error code, every SSE event, and the tuning knobs that affect what clients should expect at runtime.

For setup, environment variables, and `docker run` examples see the [README](../README.md).

## Endpoints

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/live` | none | Liveness probe. `200` while the process is running. |
| `GET` | `/health` | none | Readiness probe. `200` once the pipeline is loaded, `503` until then. |
| `GET` | `/metrics` | none | Prometheus exposition (text/plain). |
| `POST` | `/diarize` | Bearer | Submit an audio file and stream the diarization result over SSE. |

## Authentication

All authenticated endpoints expect:

```
Authorization: Bearer <key>
```

`<key>` must match one of the comma-separated entries in the `API_KEYS` environment variable. There is no rotation / expiry logic — multiple keys are simply all valid, which lets you rotate by adding a new key, deploying, then removing the old one.

Unauthenticated calls return `401`:

```json
{ "detail": { "error": "unauthorized" } }
```

## `POST /diarize`

### Request

| Part | Where | Required | Description |
| --- | --- | --- | --- |
| `Authorization: Bearer <key>` | header | yes | API key (see above). |
| `Accept: text/event-stream` | header | recommended | Signals intent; the server returns SSE regardless. |
| `file` | multipart form field | yes | Audio file (any format `torchaudio` can decode — wav, flac, mp3, m4a, …). |
| `num_speakers` | query | no | Exact number of speakers (overrides min/max). |
| `min_speakers` | query | no | Lower bound on speaker count. |
| `max_speakers` | query | no | Upper bound on speaker count. |
| `exclusive` | query | no | If `true`, return the pipeline's `exclusive_speaker_diarization` output (non-overlapping segments). Default `false`. |

### Successful response

- **Status:** `200`
- **Content-Type:** `text/event-stream`
- **Headers:** `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no`
- **Body:** a sequence of SSE frames terminated by either a `result` event (success) or an `error` event (application-level failure). See the event reference below.

The HTTP status is `200` as soon as the request is accepted onto the queue. Application-level failures during processing surface as an SSE `error` event, **not** an HTTP error, because the response has already started streaming. Clients must inspect the event stream — checking only the HTTP status is not sufficient.

### Pre-acceptance HTTP errors

The following errors are returned before the SSE stream begins, with a normal JSON body. Their shape is FastAPI's default `{"detail": ...}`.

| Status | `detail.error` | Headers | Meaning | Recommended client action |
| --- | --- | --- | --- | --- |
| `401` | `unauthorized` | — | Missing or invalid `Authorization: Bearer` header. | Fix the key. Do not retry. |
| `503` | `pipeline_not_loaded` | — | The container is up but the pipeline is still initialising (cold start, model download). | Retry with exponential backoff; `/health` will be `200` once ready. |
| `503` | `queue_full` | `Retry-After: 5` | The in-process queue has reached `MAX_QUEUE_DEPTH` (default `64`). | Honour `Retry-After`, then retry. Detail payload includes `max_queue_depth`. |
| `422` | (FastAPI validation) | — | Missing `file` field, invalid query parameter type, etc. | Fix the request; do not retry as-is. |

## SSE event reference

Each frame is:

```
event: <name>
data: <single-line JSON>

```

Frames are separated by a blank line. The stream ends after a `result` or `error` event. Clients should also handle the underlying TCP connection closing without either of those (treat as transport failure and retry).

### `event: status`

Lifecycle transitions for the job. Emitted at least twice: once on enqueue (`phase: "queued"`) and once when a worker picks the job up (`phase: "running"`).

```json
{
  "job_id": "a1b2c3...",
  "phase": "queued" | "running",
  "worker": 0
}
```

- `job_id` — opaque identifier for this request; useful for log correlation.
- `worker` — only present on `phase: "running"`; zero-based worker index.

### `event: heartbeat`

Emitted every `SSE_HEARTBEAT_SECONDS` (default `5`) while the job is queued or running. Its only purpose is to keep the connection alive through proxies and to give the client a coarse progress indicator. **It is not a guarantee of forward progress** — the worker may be deep inside a synchronous pyannote call.

```json
{
  "job_id": "a1b2c3...",
  "phase": "queued" | "running",
  "elapsed_seconds": 12.5
}
```

`elapsed_seconds` is wall time since the SSE stream started (i.e. since the upload completed and the job was enqueued), not since the worker began processing.

### `event: result`

Terminal success event. The stream closes immediately after.

```json
{
  "job_id": "a1b2c3...",
  "duration_seconds": 123.45,
  "num_speakers": 2,
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "segments": [
    { "start": 0.21, "end": 3.84, "speaker": "SPEAKER_00" }
  ],
  "processing_time_seconds": 4.2,
  "model": "pyannote/speaker-diarization-community-1",
  "pyannote_version": "<from installed pyannote.audio>"
}
```

| Field | Type | Notes |
| --- | --- | --- |
| `duration_seconds` | float | Length of the input audio, derived from `torchaudio.load`. |
| `num_speakers` | int | Distinct speaker labels in `segments`. |
| `speakers` | string[] | Sorted list of distinct speaker labels. |
| `segments` | object[] | Time-ordered speech turns. `start` < `end`, both in seconds. |
| `processing_time_seconds` | float | Wall time of the inference call only (not including upload/queue wait). |
| `model` | string | Static model id; useful for downstream auditing. |
| `pyannote_version` | string | Installed `pyannote.audio` version. |

### `event: error`

Terminal failure event during processing. The stream closes immediately after.

```json
{
  "job_id": "a1b2c3...",
  "status": 400 | 500,
  "detail": { "error": "<code>" }
}
```

| `status` | `detail.error` | Meaning | Recommended client action |
| --- | --- | --- | --- |
| `400` | `invalid_audio` | The uploaded file decoded to an unexpected shape (e.g. zero-dimensional tensor). | Validate the file locally before retrying. |
| `500` | `diarization_output_parse_failed` | The pyannote pipeline returned a structure this service did not recognise (typically a version skew). | Open an issue; retrying is unlikely to help. |
| `500` | `diarization_failed` | An unexpected runtime error from `torchaudio` / `pyannote` / `torch` (file decode failure, CUDA OOM, etc.). | Inspect server logs. Retry once after backoff; if persistent, treat as a server-side bug. |

The `status` field mirrors the HTTP status this error *would* have produced if it had happened before the stream started. It is informational; the actual HTTP status is always `200` for any response that reached the SSE phase.

## Performance and tuning

The service runs an in-process job queue: incoming requests are accepted onto an `asyncio.Queue`, and a fixed pool of workers pulls from it and runs pyannote inference in a thread (so the event loop stays free to emit heartbeats and accept new uploads).

Three environment variables control this behaviour:

### `DIARIZE_WORKERS` (default: `1`)

Number of background workers consuming the queue. **One diarization runs per worker at a time.**

- **Single GPU (the default and most common case): keep this at `1`.** pyannote saturates the GPU; two concurrent inferences contend on the same CUDA context, don't go faster, and frequently OOM on longer clips.
- **Multi-GPU host:** the current implementation always sends tensors to `cuda` (device 0). Raising `DIARIZE_WORKERS` will *not* automatically use other GPUs — both workers will fight over GPU 0. Multi-device support would need a worker→device mapping (not implemented; track in the issue tracker).
- **CPU-only image:** PyTorch already parallelises a single inference across cores via `OMP_NUM_THREADS`. Running multiple workers oversubscribes the cores unless you also reduce intra-op threads per worker.

### `MAX_QUEUE_DEPTH` (default: `64`)

Maximum number of jobs that may sit in the queue (excluding the one currently being processed). When the queue is full:

- The server rejects new `POST /diarize` calls with `503 {"detail":{"error":"queue_full","max_queue_depth":N}}` and `Retry-After: 5`.
- The check runs **before** the upload body is read, so flooding the endpoint does not waste disk I/O.
- A second check runs after the upload to close the race against other concurrent requests; the temp file is cleaned up on this path.

Sizing guidance: keep this small enough that the worst-case queue wait (`MAX_QUEUE_DEPTH * avg_processing_time / DIARIZE_WORKERS`) is shorter than your upstream client/LB timeout. With a 30 s average job and one worker, `MAX_QUEUE_DEPTH=64` implies up to ~32 minutes of queue wait for the unluckiest caller — set it lower if your clients are less patient.

### `SSE_HEARTBEAT_SECONDS` (default: `5`)

Interval between `heartbeat` frames. Must be lower than:

- The idle timeout of every proxy between the client and the service (nginx default `proxy_read_timeout` is `60s`; cloud load balancers typically `60`–`350s`).
- The client's own read timeout.

Lower values give snappier disconnect detection (the server polls `request.is_disconnected()` once per heartbeat tick) at the cost of marginally more bytes on the wire. `5` is a reasonable default for typical deployments.

## Client implementation checklist

1. **Always send `Authorization: Bearer <key>`.** Treat `401` as fatal — do not retry.
2. **Treat `503 queue_full` as backpressure.** Respect `Retry-After`. A naive retry loop without backoff will keep the queue saturated.
3. **Treat `503 pipeline_not_loaded` as a startup race.** Retry with exponential backoff; combine with a `GET /health` probe if you control the deployment lifecycle.
4. **Stream the response.** Do not buffer the full body before parsing — the heartbeats are the *point*. Use an HTTP client that exposes a byte/line stream (`fetch` + `ReadableStream`, `httpx.stream`, `requests` with `stream=True`, etc.). Browser `EventSource` cannot be used directly because it does not support `POST` with multipart bodies.
5. **Parse SSE properly.** Split on blank lines (`\n\n`), then extract `event:` and `data:` lines per block. The `data:` payload is always single-line JSON in this service.
6. **Distinguish stream events from transport errors.** A successful HTTP response that ends without a `result` or `error` event means the TCP connection was dropped (timeout, server crash, …); treat it as retryable. A `result` event means success even though the HTTP status was already `200` from the start. An `error` event means application-level failure — check the table above to decide whether to retry.
7. **Honour your own timeouts on `elapsed_seconds`.** The server will not time out long jobs; if you need a ceiling, close the connection client-side and the server will detect the disconnect on its next heartbeat tick (the worker still finishes the in-flight job, since pyannote inference is not cancellable, but no result is delivered to you).
8. **Log `job_id`.** Every event carries it; pairing it with server logs (`grep <job_id>`) is the fastest way to debug a stuck or failing request.

## Metrics (Prometheus)

`GET /metrics` exposes the following series in addition to the standard `prometheus-client` process collectors:

| Name | Type | Labels | Description |
| --- | --- | --- | --- |
| `pyannote_requests_total` | counter | `endpoint`, `status` | HTTP requests processed. SSE responses always count as `status="200"`. |
| `pyannote_diarization_duration_seconds` | histogram | — | Wall time of the pyannote inference call. |
| `pyannote_audio_duration_seconds` | histogram | — | Length of input audio. |
| `pyannote_active_requests` | gauge | — | Jobs currently being processed by a worker (does not include queued jobs). |
| `pyannote_model_loaded` | gauge | — | `1` after the pipeline is loaded, `0` during startup / shutdown. |
| `pyannote` | info | `version`, `model_id`, `torch_version`, `cuda_available` | Static build/runtime info. |
