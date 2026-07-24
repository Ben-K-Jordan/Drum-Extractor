"""Regression tests for the medium/low backlog from the open-source comparison.

Covers: constant-grid quantization, grid extension + missed-beat splitting,
tempo-range threading, tail-bar annotation, ensemble pad/attenuate/FFT modes,
the DP tab mapper, notation duration caps + voice numbering, packaging markers,
and Demucs segment threading.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drum_extractor.config import BassTranscriptionConfig, NotationConfig, QuantizeConfig, SeparationConfig
from drum_extractor.events import BassNote, DrumHit, Transcription


# --- Quantization robustness ----------------------------------------------------------

def test_constant_grid_mode_builds_uniform_beats():
    from drum_extractor.quantize import _constant_beats

    tempo, beats, downbeats = _constant_beats(
        tempo=None,
        beats=[0.5, 1.0, 1.5],
        downbeats=[1.0],
        config=QuantizeConfig(fixed_tempo=120.0, grid_mode="constant"),
        t_lo=0.0,
        t_hi=5.0,
    )
    assert tempo == 120.0
    gaps = {round(b - a, 6) for a, b in zip(beats, beats[1:])}
    assert gaps == {0.5}  # perfectly uniform at 120 BPM
    assert any(abs(d - 1.0) < 1e-9 for d in downbeats)  # anchor stays a downbeat
    assert beats[0] <= 0.0 and beats[-1] >= 5.0  # covers the onset range


def test_constant_grid_falls_back_without_tempo():
    from drum_extractor.quantize import _constant_beats

    tempo, beats, downbeats = _constant_beats(
        None, [0.0, 0.6], [0.0], QuantizeConfig(grid_mode="constant"), 0.0, 1.0
    )
    assert beats == [0.0, 0.6]  # tracked beats untouched


def test_grid_extends_past_detected_beats():
    from drum_extractor.quantize import _build_grid

    grid = _build_grid([0.0, 0.5, 1.0], QuantizeConfig(grid=16), t_lo=-0.9, t_hi=2.2)
    assert min(grid) <= -0.9 + 0.5  # extended before the first beat
    assert max(grid) >= 2.2 - 1e-6  # extended past the last beat


def test_grid_splits_missed_beat():
    from drum_extractor.quantize import _build_grid

    # Beat at 1.0 was missed by the tracker: interval 0.5->1.5 is 2x the median.
    grid = _build_grid([0.0, 0.5, 1.5, 2.0], QuantizeConfig(grid=16))
    assert any(abs(g - 1.0) < 1e-6 for g in grid)  # recovered beat slot
    assert any(abs(g - 1.125) < 1e-6 for g in grid)  # with full 16th resolution


def test_bpm_range_steers_librosa_prior(tmp_path, monkeypatch):
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
    Q._detect_librosa(tmp_path / "a.wav", QuantizeConfig(backend="librosa", min_bpm=100.0, max_bpm=200.0))
    assert captured.get("start_bpm") == 150.0  # midpoint of the range


def test_tail_hits_get_synthetic_bars():
    from drum_extractor.quantize import _annotate_bar_beat

    tr = Transcription(drum_hits=[DrumHit(0.5, "kick"), DrumHit(7.9, "snare")], tempo=120.0)
    _annotate_bar_beat(tr, [0.0, 2.0], QuantizeConfig())
    tail = tr.drum_hits[1]
    assert tail.bar == 4  # extended past the last detected downbeat (2s bars)
    assert 0.0 <= tail.beat < 4.0  # not clamped onto the final bar's end


# --- Ensemble blending ----------------------------------------------------------------

def test_attenuate_only_never_boosts(tmp_path):
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf
    from drum_extractor.ensemble import average_stems

    sr = 22050
    for name in ("a", "b"):
        sf.write(str(tmp_path / f"{name}.wav"), np.full((sr, 1), 0.1, dtype="float32"), sr)
    out = average_stems([tmp_path / "a.wav", tmp_path / "b.wav"], tmp_path / "avg.wav", align=False)
    y, _ = sf.read(str(out))
    assert abs(abs(y[100]) - 0.1) < 0.01  # quiet input stays quiet (no 0.98 boost)


def test_fft_blend_preserves_signal(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf
    from drum_extractor.ensemble import average_stems

    sr = 22050
    t = np.arange(sr) / sr
    sig = (0.5 * np.sin(2 * np.pi * 440 * t)).astype("float32")[:, None]
    for name in ("a", "b"):
        sf.write(str(tmp_path / f"{name}.wav"), sig, sr)
    out = average_stems(
        [tmp_path / "a.wav", tmp_path / "b.wav"], tmp_path / "fft.wav", align=False, algorithm="avg_fft"
    )
    y, _ = sf.read(str(out))
    assert len(y) == sr
    # Identical inputs -> FFT blend reconstructs essentially the same signal.
    interior = slice(2048, sr - 2048)
    corr = np.corrcoef(y[interior], sig[interior, 0])[0, 1]
    assert corr > 0.99


def test_unknown_algorithm_rejected(tmp_path):
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf
    from drum_extractor.ensemble import average_stems

    sf.write(str(tmp_path / "a.wav"), np.zeros((100, 1), dtype="float32"), 22050)
    with pytest.raises(ValueError):
        average_stems([tmp_path / "a.wav"], tmp_path / "o.wav", algorithm="median_of_medians")


# --- Verification-round fixes (adversarial review of this diff) -----------------------

def test_fft_blend_preserves_longer_stems_tail_regardless_of_order(tmp_path):
    """Zero-phase regression: with the SHORT stem first, the long stem's tail
    used to be resynthesized with constant zero phase (garbage)."""
    pytest.importorskip("scipy")
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf
    from drum_extractor.ensemble import average_stems

    sr = 22050
    t2 = np.arange(2 * sr) / sr
    long_sig = (0.5 * np.sin(2 * np.pi * 220 * t2)).astype("float32")[:, None]
    short_sig = long_sig[: sr]
    sf.write(str(tmp_path / "short.wav"), short_sig, sr)
    sf.write(str(tmp_path / "long.wav"), long_sig, sr)

    out = average_stems(
        [tmp_path / "short.wav", tmp_path / "long.wav"],  # short FIRST (the bad order)
        tmp_path / "blend.wav", align=False, algorithm="avg_fft",
    )
    y, _ = sf.read(str(out))
    tail = slice(sr + 4096, 2 * sr - 4096)
    corr = np.corrcoef(y[tail], long_sig[tail, 0])[0, 1]
    assert corr > 0.9, f"long stem's tail was garbled (corr={corr:.3f})"


def test_fft_blend_handles_sub_window_stems(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf
    from drum_extractor.ensemble import average_stems

    for n in (1000, 3500):  # both former crash modes (<nperseg, istft mismatch)
        for name in ("a", "b"):
            sf.write(str(tmp_path / f"{name}{n}.wav"), np.random.default_rng(0).standard_normal((n, 1)).astype("float32") * 0.1, 22050)
        out = average_stems([tmp_path / f"a{n}.wav", tmp_path / f"b{n}.wav"], tmp_path / f"o{n}.wav", align=False, algorithm="avg_fft")
        y, _ = sf.read(str(out))
        assert len(y) == n  # no ValueError, correct length


def test_channel_mismatch_raises_not_discards(tmp_path):
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf
    from drum_extractor.ensemble import average_stems
    from drum_extractor.errors import ExternalToolError

    sf.write(str(tmp_path / "st.wav"), np.zeros((1000, 2), dtype="float32"), 22050)
    sf.write(str(tmp_path / "quad.wav"), np.zeros((1000, 4), dtype="float32"), 22050)
    with pytest.raises(ExternalToolError):
        average_stems([tmp_path / "st.wav", tmp_path / "quad.wav"], tmp_path / "o.wav", align=False)


def test_constant_beats_backward_extension_is_bounded():
    from drum_extractor.quantize import _constant_beats

    _tempo, beats, _down = _constant_beats(
        None, [0.5, 1.0], [0.5], QuantizeConfig(fixed_tempo=120.0, grid_mode="constant"),
        t_lo=-60000.0, t_hi=200.0,  # one absurd stray time
    )
    assert beats[-1] >= 200.0  # the grid still reaches the actual song


def test_early_onset_never_snaps_negative(tmp_path):
    pytest.importorskip("librosa")
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf
    from drum_extractor.quantize import quantize

    # Click track whose first beat is late enough that backward extrapolation
    # creates grid slots below zero; the early hit must still clamp to >= 0.
    sr = 44100
    y = np.zeros(4 * sr, dtype="float32")
    for k in range(6):
        i = int((0.4 + k * 0.5) * sr)
        y[i : i + 300] = 0.8
    sf.write(str(tmp_path / "c.wav"), y, sr)
    tr = Transcription(drum_hits=[DrumHit(0.01, "kick"), DrumHit(0.9, "snare")])
    quantize(tr, tmp_path / "c.wav", QuantizeConfig(backend="librosa"))
    assert all(h.time >= 0.0 for h in tr.drum_hits)


def test_unknown_grid_mode_rejected(tmp_path):
    from drum_extractor.quantize import quantize

    with pytest.raises(ValueError):
        quantize(Transcription(), tmp_path / "x.wav", QuantizeConfig(grid_mode="consant"))  # typo
    with pytest.raises(ValueError):
        quantize(Transcription(), tmp_path / "x.wav", QuantizeConfig(backend="madmon"))  # typo


# --- DP tab mapper --------------------------------------------------------------------

def test_dp_tab_keeps_phrase_on_one_string():
    from drum_extractor.bass import assign_tab

    cfg = BassTranscriptionConfig()
    notes = [BassNote(i * 0.25, i * 0.25 + 0.2, p) for i, p in enumerate([45, 47, 45, 47])]
    assign_tab(notes, cfg)
    assert len({n.string for n in notes}) == 1  # no pointless string hopping


def test_dp_tab_resets_hand_after_long_rest():
    from drum_extractor.bass import assign_tab

    cfg = BassTranscriptionConfig()
    # Open low E, then a long rest, then A2: the hand is free to reposition, so
    # the mapper should pick a mid-neck fingering near the neutral position
    # rather than dragging relative to fret 0.
    notes = [BassNote(0.0, 0.5, 28), BassNote(3.0, 3.5, 45)]
    assign_tab(notes, cfg)
    assert notes[1].string is not None
    assert 2 <= notes[1].fret <= 9  # a sane mid-neck choice, not fret 12/17


def test_dp_tab_unreachable_mixed_with_playable():
    from drum_extractor.bass import assign_tab

    cfg = BassTranscriptionConfig()
    notes = [BassNote(0.0, 0.5, 20), BassNote(0.5, 1.0, 30), BassNote(1.0, 1.5, 21)]
    assign_tab(notes, cfg)
    assert notes[0].string is None and notes[2].string is None  # out of range
    assert notes[1].string is not None  # playable note still assigned


# --- Notation polish ------------------------------------------------------------------

def test_sparse_kick_capped_at_one_beat():
    pytest.importorskip("music21")
    from drum_extractor.notation import build_score

    hits = [DrumHit(0.0, "kick", bar=1, beat=0.0), DrumHit(0.0, "kick", bar=2, beat=0.0)]
    score = build_score(Transcription(drum_hits=hits, tempo=120.0), NotationConfig(), QuantizeConfig(grid=16))
    durations = [float(n.duration.quarterLength) for n in score.recurse().notes]
    assert durations and max(durations) <= 1.0 + 1e-6  # quarter, not a whole note


def test_voices_numbered_one_and_two(tmp_path):
    pytest.importorskip("music21")
    from drum_extractor.notation import transcription_to_musicxml

    hits = [
        DrumHit(0.0, "hihat_closed", bar=1, beat=0.0),
        DrumHit(0.0, "hihat_closed", bar=1, beat=1.0),
        DrumHit(0.0, "kick", bar=1, beat=0.0),
        DrumHit(0.0, "kick", bar=1, beat=2.0),
    ]
    tr = Transcription(drum_hits=hits, tempo=120.0)
    out = transcription_to_musicxml(tr, tmp_path / "v.musicxml", NotationConfig(), QuantizeConfig())
    xml = out.read_text()
    assert "<voice>1</voice>" in xml  # hands
    assert "<voice>2</voice>" in xml  # feet
    assert "<voice>0</voice>" not in xml


# --- Packaging / plumbing -------------------------------------------------------------

def test_py_typed_marker_ships():
    assert (Path(__file__).parent.parent / "drum_extractor" / "py.typed").exists()


def test_segment_flag_reaches_demucs_cli(tmp_path, monkeypatch):
    import subprocess

    captured = {}

    class _Result:
        returncode = 0

    def fake_run(cmd, check=True, **kwargs):
        captured["cmd"] = cmd
        # Write the layout the real CLI produces, so the exit-0-but-no-files
        # guard (a separate audited fix) doesn't fire on this happy path.
        track_dir = tmp_path / "out" / "htdemucs_ft" / "song"
        track_dir.mkdir(parents=True, exist_ok=True)
        for name in ("drums", "bass"):
            (track_dir / f"{name}.wav").write_bytes(b"RIFF")
        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    from drum_extractor.separation import _separate_cli

    stems = _separate_cli(tmp_path / "song.wav", tmp_path / "out", SeparationConfig(segment=12), "cpu")
    assert "--segment" in captured["cmd"]
    assert "12" in captured["cmd"]
    assert stems.drums and stems.drums.exists()


def test_cli_exit_zero_but_no_stems_raises(tmp_path, monkeypatch):
    """Demucs exiting 0 without writing stems must fail loudly, not return Stems()."""
    import subprocess

    import pytest

    from drum_extractor.errors import ExternalToolError
    from drum_extractor.separation import _separate_cli

    class _Result:
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda cmd, check=True, **k: _Result())
    with pytest.raises(ExternalToolError, match="wrote no"):
        _separate_cli(tmp_path / "song.wav", tmp_path / "out", SeparationConfig(), "cpu")


def test_cli_flags_parse():
    from drum_extractor.cli import build_parser

    args = build_parser().parse_args(
        ["run", "s.mp3", "--ensemble-algorithm", "avg_fft", "--segment", "10"]
    )
    assert args.ensemble_algorithm == "avg_fft"
    assert args.segment == 10
