"""Repeated low-string attacks that the note model merges get split back out."""

from __future__ import annotations

import numpy as np
import pytest

from drum_extractor.bass import split_reattacks
from drum_extractor.events import BassNote

librosa = pytest.importorskip("librosa")
sf = pytest.importorskip("soundfile")

SR = 22050
F1, BB1 = 29, 34
EIGHTH = 60.0 / 117 / 2


def _hz(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _chug(midi: int, dur: float) -> np.ndarray:
    """A saturated, decaying pick attack, like a palm-muted bass note."""
    t = np.arange(int(dur * SR)) / SR
    wave = np.sign(np.sin(2 * np.pi * _hz(midi) * t)) * 0.35 + np.sin(2 * np.pi * _hz(midi) * t) * 0.65
    return np.tanh(1.8 * wave * np.minimum(1, t / 0.005) * np.exp(-t * 3.5))


def _render(events, total: float, path) -> str:
    y = np.zeros(int(total * SR))
    for start, midi, dur in events:
        snd = _chug(midi, dur)
        i = int(start * SR)
        y[i:i + len(snd)] += snd[: len(y) - i]
    sf.write(str(path), (0.8 * y / np.abs(y).max()).astype("float32"), SR)
    return str(path)


def test_merged_e_string_chugs_are_split(tmp_path):
    starts = [0.1, 0.1 + EIGHTH, 0.1 + 3 * EIGHTH]  # eighths 0, 1 and 3
    wav = _render([(s, F1, EIGHTH * 1.8) for s in starts], 1.4, tmp_path / "chug.wav")
    merged = [BassNote(start=0.1, end=1.2, pitch=F1, velocity=100)]
    out = split_reattacks(wav, merged)
    assert [n.pitch for n in out] == [F1, F1, F1]
    assert [round(n.start, 2) for n in out] == pytest.approx(starts, abs=0.03)
    assert out[-1].end == pytest.approx(1.2)


def test_held_note_with_vibrato_stays_whole(tmp_path):
    t = np.arange(int(1.5 * SR)) / SR
    hz = _hz(F1) * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))
    y = np.sin(2 * np.pi * np.cumsum(hz) / SR) * np.minimum(1, t / 0.005) * np.exp(-t * 0.8)
    sf.write(str(tmp_path / "held.wav"), (0.8 * y).astype("float32"), SR)
    held = [BassNote(start=0.0, end=1.4, pitch=F1, velocity=100)]
    assert len(split_reattacks(str(tmp_path / "held.wav"), held)) == 1


def test_pitch_change_under_an_overlapping_note_leaves_no_fragment(tmp_path):
    # F1 then Bb1; the note model starts Bb1 early and lets F1 ring into it.
    wav = _render([(0.1, F1, 0.6), (0.6, BB1, 0.6)], 1.4, tmp_path / "change.wav")
    notes = [BassNote(start=0.1, end=0.7, pitch=F1, velocity=100),
             BassNote(start=0.55, end=1.15, pitch=BB1, velocity=100)]
    out = split_reattacks(wav, notes)
    assert [(n.pitch, round(n.start, 2)) for n in out] == [(F1, 0.1), (BB1, 0.55)]
