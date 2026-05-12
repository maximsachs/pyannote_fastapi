from __future__ import annotations

import io
import json
import wave

from fastapi.testclient import TestClient

from main import app


def _parse_sse(stream_text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for raw_block in stream_text.split("\n\n"):
        block = raw_block.strip()
        if not block:
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if data_lines:
            events.append((event_name, json.loads("\n".join(data_lines))))
    return events


def _silent_wav_bytes(duration_seconds: float = 0.2, sample_rate: int = 16000) -> bytes:
    n = int(duration_seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


def test_live_and_health() -> None:
    with TestClient(app) as client:
        r = client.get("/live")
        assert r.status_code == 200
        assert r.json().get("status") == "ok"
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json().get("status") == "ready"


def test_metrics_prometheus_text() -> None:
    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.text
        assert "pyannote_requests_total" in body


def test_diarize_requires_auth() -> None:
    with TestClient(app) as client:
        audio = _silent_wav_bytes()
        r = client.post("/diarize", files={"file": ("test.wav", audio, "audio/wav")})
        assert r.status_code == 401


def test_diarize_rejects_wrong_bearer_token() -> None:
    with TestClient(app) as client:
        audio = _silent_wav_bytes()
        r = client.post(
            "/diarize",
            headers={"Authorization": "Bearer not-a-real-key"},
            files={"file": ("test.wav", audio, "audio/wav")},
        )
        assert r.status_code == 401


def test_diarize_success_with_bearer_token() -> None:
    with TestClient(app) as client:
        audio = _silent_wav_bytes()
        r = client.post(
            "/diarize",
            headers={"Authorization": "Bearer test-integration-key"},
            files={"file": ("test.wav", audio, "audio/wav")},
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(r.text)
        event_names = [name for name, _ in events]
        assert "status" in event_names
        assert "result" in event_names
        result_payload = next(data for name, data in events if name == "result")
        assert "segments" in result_payload
        assert result_payload["num_speakers"] >= 1
        assert "job_id" in result_payload
