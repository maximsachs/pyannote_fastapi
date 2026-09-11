from __future__ import annotations

import io
import json
import wave

import pytest
from fastapi.testclient import TestClient
from pyannote.core import Annotation, Segment
from pydantic import ValidationError

import main
from main import _DiarizationParams, app


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


def test_diarization_params_reject_non_positive_counts() -> None:
    with pytest.raises(ValidationError):
        _DiarizationParams(num_speakers=0)
    with pytest.raises(ValidationError):
        _DiarizationParams(min_speakers=-1)
    with pytest.raises(ValidationError):
        _DiarizationParams(min_speakers=3, max_speakers=1)


def test_diarization_params_allow_num_speakers_with_bounds() -> None:
    params = _DiarizationParams(num_speakers=2, min_speakers=1, max_speakers=5)
    assert params.num_speakers == 2
    assert params.min_speakers == 1
    assert params.max_speakers == 5


def test_diarize_rejects_invalid_speaker_count_query() -> None:
    audio = _silent_wav_bytes()
    headers = {"Authorization": "Bearer test-integration-key"}
    with TestClient(app) as client:
        zero = client.post(
            "/diarize",
            headers=headers,
            params={"num_speakers": 0},
            files={"file": ("test.wav", audio, "audio/wav")},
        )
        assert zero.status_code == 422
        inverted = client.post(
            "/diarize",
            headers=headers,
            params={"min_speakers": 4, "max_speakers": 2},
            files={"file": ("test.wav", audio, "audio/wav")},
        )
        assert inverted.status_code == 422


def test_diarize_forwards_speaker_hints_and_reports_detected_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _SpyPipeline:
        def __call__(self, *_args: object, **kwargs: object) -> dict[str, Annotation]:
            captured.clear()
            captured.update(kwargs)
            ann = Annotation()
            ann[Segment(0.0, 0.4)] = "SPEAKER_00"
            ann[Segment(0.5, 0.9)] = "SPEAKER_01"
            return {"speaker_diarization": ann}

    with TestClient(app) as client:
        monkeypatch.setattr(main, "_pipeline", _SpyPipeline())
        r = client.post(
            "/diarize",
            headers={"Authorization": "Bearer test-integration-key"},
            params={"num_speakers": 4, "min_speakers": 2, "max_speakers": 6},
            files={"file": ("test.wav", _silent_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 200
        result_payload = next(data for name, data in _parse_sse(r.text) if name == "result")

    assert captured == {
        "num_speakers": 4,
        "min_speakers": 2,
        "max_speakers": 6,
    }
    assert result_payload["num_speakers"] == 2
    assert result_payload["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
