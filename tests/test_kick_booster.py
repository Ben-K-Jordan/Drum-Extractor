"""Verify the double-kick booster recovers fast kicks. Skips without librosa/scipy."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("librosa")
pytest.importorskip("scipy")
pytest.importorskip("soundfile")

import soundfile as sf  # noqa: E402

from drum_extractor.events import DrumHit  # noqa: E402
from drum_extractor.gm_drum_map import KICK  # noqa: E402
from drum_extractor.kick import boost_double_kick, detect_kick_onsets  # noqa: E402


def _fast_double_kick(path, n_kicks=16, gap_s=0.075, sr=44100):
    """n_kicks at gap_s spacing (0.075s ~= 200 BPM sixteenths)."""
    total = int((n_kicks * gap_s + 0.3) * sr)
    audio = np.zeros(total)
    for k in range(n_kicks):
        t0 = int(k * gap_s * sr)
        t = np.linspace(0, 0.06, int(0.06 * sr))
        freq = 120 * np.exp(-t * 22) + 45
        snd = 0.9 * np.exp(-t * 35) * np.sin(2 * np.pi * np.cumsum(freq) / sr)
        j = min(t0 + len(snd), total)
        audio[t0:j] += snd[: j - t0]
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.9
    sf.write(str(path), audio.astype(np.float32), sr)
    return n_kicks


def test_detects_most_fast_kicks(tmp_path):
    stem = tmp_path / "kick.wav"
    n = _fast_double_kick(stem, n_kicks=16, gap_s=0.075)
    onsets = detect_kick_onsets(stem, min_gap_ms=30)
    # Should recover most of the fast kicks (allow a little slack at the edges).
    assert len(onsets) >= n - 3, f"only found {len(onsets)}/{n} fast kicks"


def test_booster_does_not_decimate_fast_kicks(tmp_path):
    """Regression (audit M3): recovered kicks must not cannibalize each other.

    With the old insort-based dedup, a 40ms merge window dropped genuine kicks
    ~30-45ms apart. From an empty starting set every detected kick must survive.
    """
    stem = tmp_path / "fast.wav"
    n = _fast_double_kick(stem, n_kicks=16, gap_s=0.045)  # ~200 BPM 16ths
    detected = detect_kick_onsets(stem, min_gap_ms=30)
    boosted = boost_double_kick([], stem)  # no pre-existing kicks
    kept = sum(1 for h in boosted if h.instrument == KICK)
    assert kept == len(detected), f"decimated: kept {kept} of {len(detected)} detected"
    assert kept >= n - 4  # detector recovers most of the 16 real kicks


def test_booster_adds_missing_kicks_without_doubling(tmp_path):
    stem = tmp_path / "kick.wav"
    _fast_double_kick(stem, n_kicks=16, gap_s=0.075)
    # Simulate a transcriber that caught only every 4th kick.
    sparse = [DrumHit(time=k * 0.075, instrument=KICK) for k in range(0, 16, 4)]
    boosted = boost_double_kick(sparse, stem)
    kicks = [h for h in boosted if h.instrument == KICK]
    assert len(kicks) > len(sparse)  # recovered extra kicks
    # No two kicks closer than the merge window collapsed into duplicates.
    times = sorted(h.time for h in kicks)
    assert all(b - a > 0.02 for a, b in zip(times, times[1:]))
