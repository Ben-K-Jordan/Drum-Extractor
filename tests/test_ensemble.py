"""Tests for ensemble stem averaging. Skips without numpy/soundfile."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpy")
pytest.importorskip("soundfile")

import soundfile as sf  # noqa: E402

from drum_extractor.ensemble import average_stems  # noqa: E402


def test_average_of_two_signals(tmp_path):
    sr = 22050
    a = np.ones((sr, 1), dtype="float32") * 0.4
    b = np.ones((sr, 1), dtype="float32") * 0.8
    sf.write(str(tmp_path / "a.wav"), a, sr)
    sf.write(str(tmp_path / "b.wav"), b, sr)

    out = average_stems([tmp_path / "a.wav", tmp_path / "b.wav"], tmp_path / "avg.wav")
    y, out_sr = sf.read(str(out))
    assert out_sr == sr
    # Mean of 0.4 and 0.8 is 0.6; after peak-normalisation a constant stays constant.
    assert np.allclose(y, y[0])  # constant signal preserved
    assert 0.5 < abs(y[0]) <= 0.98


def test_average_aligns_to_shortest(tmp_path):
    sr = 22050
    sf.write(str(tmp_path / "long.wav"), np.zeros((sr, 1), dtype="float32"), sr)
    sf.write(str(tmp_path / "short.wav"), np.zeros((sr // 2, 1), dtype="float32"), sr)
    out = average_stems([tmp_path / "long.wav", tmp_path / "short.wav"], tmp_path / "avg.wav")
    y, _ = sf.read(str(out))
    assert len(y) == sr // 2  # trimmed to the shorter input


def test_average_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        average_stems([], tmp_path / "x.wav")
