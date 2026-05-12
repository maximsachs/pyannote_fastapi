from __future__ import annotations

import os

import pytest
import torch

# Import-time configuration for tests (must run before `main` is imported).
os.environ["API_KEYS"] = "test-integration-key"
os.environ["PYANNOTE_TESTING"] = "1"


@pytest.fixture(autouse=True)
def _patch_torchaudio_load(monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    def _fake_load(_path: str) -> tuple[torch.Tensor, int]:
        return torch.zeros(1, 3200), 16000

    monkeypatch.setattr(main.torchaudio, "load", _fake_load)
