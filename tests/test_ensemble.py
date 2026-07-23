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


def test_average_pads_to_longest(tmp_path):
    # UVR-style: the shorter stem is zero-padded, preserving the longer's tail
    # (previously we truncated to the shortest and silently discarded it).
    sr = 22050
    long = np.zeros((sr, 1), dtype="float32")
    long[-100:] = 0.5  # audible tail only the longer stem has
    sf.write(str(tmp_path / "long.wav"), long, sr)
    sf.write(str(tmp_path / "short.wav"), np.zeros((sr // 2, 1), dtype="float32"), sr)
    out = average_stems([tmp_path / "long.wav", tmp_path / "short.wav"], tmp_path / "avg.wav")
    y, _ = sf.read(str(out))
    assert len(y) == sr  # padded to the longer input
    assert abs(y[-50]) > 0.1  # the tail survived (halved by averaging, not dropped)


def test_average_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        average_stems([], tmp_path / "x.wav")
