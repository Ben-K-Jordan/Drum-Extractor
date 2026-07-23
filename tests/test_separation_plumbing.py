"""Verify the Demucs tensor->file plumbing against the real library.

We can't download model weights in CI/sandboxes, but we can confirm that the
code which turns Demucs's output dict into stem files works with the real
``demucs.api.save_audio`` — the part most likely to break on an API change.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("demucs")
pytest.importorskip("soundfile")

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from drum_extractor.config import SeparationConfig  # noqa: E402
from drum_extractor.separation import _save_stems  # noqa: E402


def test_save_stems_writes_only_requested(tmp_path):
    sr = 22050
    separated = {name: torch.randn(2, sr) * 0.1 for name in ("drums", "bass", "other", "vocals")}

    stems = _save_stems(separated, tmp_path, SeparationConfig(stems=("drums", "bass")), sr)

    # Requested stems written; others skipped.
    assert stems.drums and stems.drums.exists()
    assert stems.bass and stems.bass.exists()
    assert stems.other is None
    assert stems.vocals is None

    # Files are valid, correctly-shaped audio.
    y, out_sr = sf.read(str(stems.drums))
    assert out_sr == sr
    assert y.shape[0] == sr  # samples
