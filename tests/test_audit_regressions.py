"""Regression tests for the defects found in the adversarial audit.

Each test pins a specific fixed bug so it cannot silently return. Tests needing
audio/notation deps skip cleanly without them.
"""

from __future__ import annotations

import pytest

from drum_extractor.config import (
    BassTranscriptionConfig,
    DrumTranscriptionConfig,
    NotationConfig,
    QuantizeConfig,
)
from drum_extractor.events import BassNote, DrumHit, Transcription


# --- H1: anacrusis / pre-downbeat hits must not produce negative beats ---------------

def test_h1_anacrusis_no_negative_beats():
    from drum_extractor.quantize import _annotate_bar_beat

    tr = Transcription(drum_hits=[DrumHit(t, "kick") for t in [0.0, 0.5, 1.0, 1.5]], tempo=120.0)
    # madmom-style: first downbeat is later than the first beat (song starts on a pickup).
    _annotate_bar_beat(tr, [1.0, 3.0, 5.0, 7.0], QuantizeConfig(backend="madmom"))
    assert all(h.beat is not None and h.beat >= 0 for h in tr.drum_hits)
    assert all(h.bar is not None and h.bar >= 1 for h in tr.drum_hits)


# --- M2: last-bar beat positions must be tempo-correct, not assume 120 BPM ------------

def test_m2_last_bar_beats_at_non_120_bpm():
    from drum_extractor.quantize import _annotate_bar_beat

    # 150 BPM 4/4 -> bar = 1.6s. Downbeats [0.0, 1.6]; hits on beats 1-4 of the last bar.
    tr = Transcription(drum_hits=[DrumHit(t, "snare") for t in [1.6, 2.0, 2.4, 2.8]], tempo=150.0)
    _annotate_bar_beat(tr, [0.0, 1.6], QuantizeConfig(backend="librosa"))
    beats = [h.beat for h in tr.drum_hits]
    assert beats == [0.0, 1.0, 2.0, 3.0], beats


# --- H2 + M4: notation stays well-formed (no overfull bars, correct meter) ------------

def test_h2_no_overfull_measures_on_barline_crossing():
    pytest.importorskip("music21")
    from drum_extractor.notation import build_score

    # A kick on beat 3 of bar 1 with the next kick a bar later would, unfixed,
    # sustain across the barline and overfill measure 1.
    hits = [DrumHit(0.0, "kick", bar=1, beat=2.0), DrumHit(0.0, "kick", bar=2, beat=1.0)]
    score = build_score(Transcription(drum_hits=hits, tempo=120.0), NotationConfig(), QuantizeConfig(grid=16))
    for m in score.recurse().getElementsByClass("Measure"):
        assert m.duration.quarterLength <= m.barDuration.quarterLength + 1e-6


def test_m4_non_quarter_meter_measure_count():
    pytest.importorskip("music21")
    from drum_extractor.notation import build_score

    hits = [
        DrumHit(0.0, "kick", bar=1, beat=0.0), DrumHit(0.0, "snare", bar=1, beat=3.0),
        DrumHit(0.0, "kick", bar=2, beat=0.0), DrumHit(0.0, "snare", bar=2, beat=3.0),
    ]
    tr = Transcription(drum_hits=hits, tempo=120.0, time_signature=(6, 8))
    score = build_score(tr, NotationConfig(), QuantizeConfig(beats_per_bar=6, beat_unit=8, grid=16))
    measures = list(score.recurse().getElementsByClass("Measure"))
    assert len(measures) == 2, f"6/8 produced {len(measures)} measures, expected 2"


# --- M1: onset classifier must not fabricate absent instruments -----------------------

def test_m1_no_phantom_kicks_on_hihat_only():
    pytest.importorskip("librosa")
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf
    from drum_extractor.drums import _transcribe_onsets

    sr = 44100
    rng = np.random.default_rng(0)
    n = int(4 * sr)
    audio = np.zeros(n)
    for k in range(16):  # hi-hat-only: high-frequency bursts, no kick/snare energy
        t0 = int(k * 0.25 * sr)
        tt = np.linspace(0, 0.05, int(0.05 * sr))
        hh = 0.3 * np.exp(-tt * 80) * np.diff(rng.standard_normal(len(tt)), prepend=0)
        audio[t0 : t0 + len(hh)] += hh
    audio = audio / (np.abs(audio).max() + 1e-9) * 0.9

    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "hh.wav")
        sf.write(p, audio.astype(np.float32), sr)
        hits = _transcribe_onsets(p, DrumTranscriptionConfig())
    kinds = {h.instrument for h in hits}
    assert "kick" not in kinds, "fabricated phantom kicks on hi-hat-only audio"
    assert "snare" not in kinds, "fabricated phantom snares on hi-hat-only audio"
    assert "hihat_closed" in kinds


# --- H3: CREPE refinement must track pitch, not collapse below its lowest bin ---------

def test_h3_crepe_refinement_preserves_correct_low_note(tmp_path):
    pytest.importorskip("torchcrepe")
    pytest.importorskip("librosa")
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf
    from drum_extractor.bass import _refine_octaves_with_crepe

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = np.sin(2 * np.pi * 61.74 * t).astype(np.float32)  # B1 = MIDI 35
    p = tmp_path / "b1.wav"
    sf.write(str(p), tone, sr)

    # A correctly-transcribed B1: with fmin=30 (old bug) CREPE collapsed to ~MIDI 23
    # and this note was wrongly dropped an octave to 23. It must now stay 35.
    notes = [BassNote(0.0, 1.0, 35)]
    _refine_octaves_with_crepe(p, notes, min_frequency=32.7)
    assert notes[0].pitch == 35, f"correct low note was corrupted to {notes[0].pitch}"


# --- M5: notate <file>.mid must carry the MIDI's real tempo ---------------------------

def test_m5_read_drum_tempo(tmp_path):
    pytest.importorskip("pretty_midi")
    from drum_extractor.midi_io import read_drum_tempo, write_drum_midi

    p = write_drum_midi([DrumHit(i * 0.3333, "kick") for i in range(8)], tmp_path / "t.mid", tempo=180)
    tempo = read_drum_tempo(p)
    assert tempo is not None and abs(tempo - 180) < 1.0


# --- L7: MissingDependencyError for a core dep points at the package, not a fake extra -

def test_l7_pretty_midi_hint_is_valid():
    from drum_extractor.errors import MissingDependencyError

    msg = str(MissingDependencyError("MIDI export/import", "pretty_midi"))
    assert "pip install pretty_midi" in msg
    assert "[midi]" not in msg  # the non-existent extra must not be suggested


# --- L5 + L6: ASCII tab alignment and out-of-range notes ------------------------------

def test_l5_tab_alignment_with_accidental_tuning():
    from drum_extractor.bass import assign_tab, render_ascii_tab

    cfg = BassTranscriptionConfig(tuning=(30, 35, 40, 45))  # F# B E A -> a 2-char label
    notes = [BassNote(0.0, 0.5, 40), BassNote(0.5, 1.0, 47)]
    assign_tab(notes, cfg)
    lines = render_ascii_tab(notes, cfg).splitlines()
    # All string-label prefixes (up to the '|') must be the same width.
    prefixes = [ln.split("|")[0] for ln in lines]
    assert len({len(p) for p in prefixes}) == 1, prefixes


def test_l6_out_of_range_note_marked_not_dropped():
    from drum_extractor.bass import assign_tab, render_ascii_tab

    cfg = BassTranscriptionConfig()  # lowest string E1 = MIDI 28
    notes = [BassNote(0.0, 0.5, 20), BassNote(0.5, 1.0, 28)]  # 20 is below the fretboard
    assign_tab(notes, cfg)
    assert notes[0].string is None  # genuinely unplayable
    tab = render_ascii_tab(notes, cfg)
    assert "x" in tab  # but rendered as 'x', not silently dropped


# --- M6: an unexpected error in an optional stage must not abort the run --------------

def test_m6_pipeline_survives_notation_error(tmp_path, monkeypatch):
    pytest.importorskip("librosa")
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf
    from drum_extractor.config import PipelineConfig
    from drum_extractor import pipeline as P

    song_dir = tmp_path / "song"
    (song_dir / "stems").mkdir(parents=True)
    sr = 44100
    y = np.zeros(int(2 * sr), dtype="float32")
    for k in range(8):
        i = int(k * 0.25 * sr)
        y[i : i + 200] += 0.5
    sf.write(str(song_dir / "stems" / "drums.wav"), y, sr)

    # Force the notation stage to raise a non-DrumExtractorError.
    def boom(*a, **k):
        raise RuntimeError("simulated notation failure")

    monkeypatch.setattr(P, "_do_notation", boom)

    config = PipelineConfig(
        output_dir=tmp_path,
        do_separation=False,
        drums=DrumTranscriptionConfig(backend="onset"),
        bass=BassTranscriptionConfig(backend="none"),
        quantize=QuantizeConfig(backend="librosa"),
        do_sonify=False,
    )
    # Must not raise, and must still have persisted the transcription IR.
    result = P.run_pipeline(song_dir / "song.wav", config)
    assert any("notation" in w for w in result.warnings)
    assert (song_dir / "transcription.json").exists()
