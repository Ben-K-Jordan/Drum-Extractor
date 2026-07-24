"""Tests for the sonification correction aid. Skips without numpy/soundfile."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("soundfile")

import soundfile as sf  # noqa: E402

from drum_extractor.events import DrumHit  # noqa: E402
from drum_extractor.sonify import sonify_drums, write_onset_csv  # noqa: E402


def _hits():
    return [
        DrumHit(0.0, "kick", 120),
        DrumHit(0.0, "crash", 110),
        DrumHit(0.5, "snare", 90),
        DrumHit(0.25, "hihat_closed", 70),
        DrumHit(1.0, "kick", 100),
    ]


def test_sonify_writes_playable_audio(tmp_path):
    out = sonify_drums(_hits(), tmp_path / "son.wav")
    assert out.exists()
    y, sr = sf.read(str(out))
    assert sr == 44100
    # Duration covers the last hit (1.0s) plus tail.
    assert len(y) / sr > 1.0
    assert abs(y).max() > 0.0  # non-silent


def test_sonify_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        sonify_drums([], tmp_path / "empty.wav")


def test_onset_csv(tmp_path):
    out = write_onset_csv(_hits(), tmp_path / "onsets.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "time_sec,instrument,velocity_midi"
    assert len(lines) == 1 + len(_hits())
    # Sorted by time; first data row is a t=0 hit.
    assert lines[1].startswith("0.0000,")
