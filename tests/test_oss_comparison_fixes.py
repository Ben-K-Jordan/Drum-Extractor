"""Regression tests for fixes from the open-source code comparison.

Each pins a specific issue found by diffing our code against the upstream
projects (ADTOF, Demucs/UVR, madmom/librosa, torchcrepe, basic-pitch).
"""

from __future__ import annotations

import pytest

from drum_extractor.config import DrumTranscriptionConfig, QuantizeConfig
from drum_extractor.events import DrumHit


# --- ADTOF: the default command must target a real tool, and output detection ---------
# must dispatch by CONTENT (MIDI vs text), not a hardcoded filename.

def test_adtof_default_command_is_not_python_m_adtof():
    # `python -m adtof` matches no real ADTOF install; the default must invoke the
    # real `adtof` console script (ADTOF-pytorch).
    cmd = DrumTranscriptionConfig().adtof_command
    assert cmd[0] == "adtof"
    assert "-m" not in cmd
    assert "{input}" in cmd and "{output}" in cmd


def test_adtof_output_detection_by_content(tmp_path):
    pytest.importorskip("pretty_midi")
    from drum_extractor.drums import _read_adtof_output
    from drum_extractor.midi_io import write_drum_midi

    stem = tmp_path / "drums.wav"
    stem.write_bytes(b"x")

    # MIDI content -> read as MIDI.
    mp = tmp_path / "drums.adtof.mid"
    write_drum_midi([DrumHit(0.0, "kick"), DrumHit(0.5, "snare")], mp, tempo=120)
    hits = _read_adtof_output(mp, stem, 96)
    assert [h.instrument for h in hits] == ["kick", "snare"]

    # Text content even in a .mid-named file -> parsed as onset text, not garbage.
    mp.write_text("0.0\t36\n0.5\t38\n0.75\t42\n")
    hits = _read_adtof_output(mp, stem, 96)
    assert len(hits) == 3


# --- fixed_tempo must be PASSED INTO tracking, not merely relabel the report ----------

def test_fixed_tempo_is_passed_to_beat_tracker(tmp_path, monkeypatch):
    pytest.importorskip("librosa")
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf
    import librosa
    from drum_extractor import quantize as Q

    sf.write(str(tmp_path / "a.wav"), np.zeros(44100, dtype="float32"), 44100)

    captured = {}
    real = librosa.beat.beat_track

    def spy(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(librosa.beat, "beat_track", spy)
    Q._detect_librosa(tmp_path / "a.wav", QuantizeConfig(backend="librosa", fixed_tempo=137.0))
    assert captured.get("bpm") == 137.0  # tempo threaded into the tracker


# --- Ensemble: stems must be time-aligned before averaging (else transients cancel) ---

def test_best_lag_recovers_known_shift():
    pytest.importorskip("scipy")
    import numpy as np
    from drum_extractor.ensemble import _best_lag

    n = 44100
    a = np.zeros(n)
    a[1000:1050] = 1.0
    b = np.zeros(n)
    b[1020:1070] = 1.0  # same transient, +20 samples later
    lag = _best_lag(a, b, int(0.05 * 44100))
    # np.roll(b, lag) must move b's transient back onto a's -> lag == -20.
    assert abs(lag + 20) <= 2
    assert np.argmax(np.roll(b, lag)) - np.argmax(a) == pytest.approx(0, abs=2)


def test_average_stems_alignment_reduces_cancellation(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf
    from drum_extractor.ensemble import average_stems

    sr = 44100
    n = sr
    # A short high-frequency burst; B is the same burst shifted half a period so a
    # naive (unaligned) average partially cancels it.
    t = np.arange(400) / sr
    burst = np.sin(2 * np.pi * 2000 * t)
    a = np.zeros(n)
    a[1000:1400] = burst
    shift = int(sr / 2000 / 2)  # half period
    b = np.zeros(n)
    b[1000 + shift : 1400 + shift] = burst
    sf.write(str(tmp_path / "a.wav"), a.astype("float32"), sr)
    sf.write(str(tmp_path / "b.wav"), b.astype("float32"), sr)

    aligned = average_stems([tmp_path / "a.wav", tmp_path / "b.wav"], tmp_path / "al.wav", align=True)
    naive = average_stems([tmp_path / "a.wav", tmp_path / "b.wav"], tmp_path / "na.wav", align=False)
    ya, _ = sf.read(str(aligned))
    yn, _ = sf.read(str(naive))
    # Aligned preserves more burst energy than the phase-cancelled naive average.
    assert float(np.sum(ya ** 2)) > float(np.sum(yn ** 2))
