from __future__ import annotations

import os

import pytest
import torch

# Import-time configuration for tests (must run before `main` is imported).
os.environ["API_KEYS"] = "test-integration-key"
os.environ["PYANNOTE_TESTING"] = "1"
# Test-friendly rate limits: tight enough to exercise the limiter from
# dedicated tests, loose enough that ordinary functional tests do not trip
# them. The autouse fixture below resets state between tests.
os.environ.setdefault("RATE_LIMIT_DIARIZE", "3/minute")
os.environ.setdefault("RATE_LIMIT_DIARIZE_IP", "5/minute")
os.environ.setdefault("RATE_LIMIT_LIVE", "10/minute")
os.environ.setdefault("RATE_LIMIT_HEALTH", "10/minute")
os.environ.setdefault("RATE_LIMIT_METRICS", "10/minute")
os.environ.setdefault("AUTH_FAIL_DELAY_SECONDS", "0")


@pytest.fixture(autouse=True)
def _patch_torchaudio_load(monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    def _fake_load(_path: str) -> tuple[torch.Tensor, int]:
        return torch.zeros(1, 3200), 16000

    monkeypatch.setattr(main.torchaudio, "load", _fake_load)


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    """Reset the slowapi storage between tests so per-test limits are isolated."""
    import main

    main.limiter.reset()
    yield
    main.limiter.reset()
