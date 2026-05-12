from __future__ import annotations

import io
import wave

from fastapi.testclient import TestClient

from main import app


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


def test_diarize_success_with_api_key() -> None:
    with TestClient(app) as client:
        audio = _silent_wav_bytes()
        r = client.post(
            "/diarize",
            headers={"X-API-Key": "test-integration-key"},
            files={"file": ("test.wav", audio, "audio/wav")},
        )
        assert r.status_code == 200
        payload = r.json()
        assert "segments" in payload
        assert payload["num_speakers"] >= 1
