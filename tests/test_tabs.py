"""Tests for the polyphonic tab engine and the guitar option."""

from __future__ import annotations

import pytest

from drum_extractor.config import GuitarTranscriptionConfig, PipelineConfig
from drum_extractor.events import BassNote
from drum_extractor.tabs import assign_frets, group_chords, render_ascii_tab

GUITAR = GuitarTranscriptionConfig()


def _chord(pitches, start=0.0, dur=0.5):
    return [BassNote(start, start + dur, p) for p in pitches]


# --- chord assignment ------------------------------------------------------------------

def test_power_chord_lands_on_distinct_strings():
    notes = _chord([40, 47, 52])  # E5: root, fifth, octave
    assign_frets(notes, GUITAR.tuning, GUITAR.frets)
    got = sorted((n.string, n.fret) for n in notes)
    assert got == [(0, 0), (1, 2), (2, 2)]  # the classic open-E5 voicing


def test_chord_strings_are_unique_and_uncrossed():
    notes = _chord([45, 52, 57, 61])  # A-shape chord
    assign_frets(notes, GUITAR.tuning, GUITAR.frets)
    placed = [(n.pitch, n.string) for n in notes if n.string is not None]
    strings = [s for _, s in placed]
    assert len(strings) == len(set(strings))  # one note per string
    ordered = sorted(placed)  # higher pitch on higher string (no crossings)
    assert [s for _, s in ordered] == sorted(strings)


def test_oversized_chord_keeps_highest_notes():
    notes = _chord([40, 45, 50, 55, 59, 64, 69])  # 7 notes, 6 strings
    unplaceable = assign_frets(notes, GUITAR.tuning, GUITAR.frets)
    assert unplaceable == 1
    dropped = [n for n in notes if n.string is None]
    assert len(dropped) == 1 and dropped[0].pitch == 40  # lowest sacrificed, melody kept


def test_chord_grouping_by_onset():
    notes = [BassNote(0.0, 0.5, 40), BassNote(0.01, 0.5, 47), BassNote(0.30, 0.6, 45)]
    groups = group_chords(notes)
    assert [len(g) for g in groups] == [2, 1]


def test_wide_span_voicings_rejected():
    # C3 + E4: only voicings within a 5-fret hand span survive.
    notes = _chord([48, 64])
    assign_frets(notes, GUITAR.tuning, GUITAR.frets)
    fretted = [n.fret for n in notes if n.fret and n.fret > 0]
    if len(fretted) == 2:
        assert abs(fretted[0] - fretted[1]) <= 5


# --- rendering ---------------------------------------------------------------------------

def test_chord_shares_one_column():
    notes = _chord([40, 47, 52])
    assign_frets(notes, GUITAR.tuning, GUITAR.frets)
    tab = render_ascii_tab(notes, GUITAR.tuning)
    lines = tab.splitlines()
    assert len(lines) == 6  # one system, six strings
    # All three frets sit at the same column position (same chord column).
    cols = {ln.index("0") if "0" in ln else ln.index("2") for ln in lines if any(c.isdigit() for c in ln)}
    assert len(cols) == 1


def test_long_line_wraps_into_systems():
    notes = []
    for i in range(60):
        notes.append(BassNote(i * 0.25, i * 0.25 + 0.2, 45))
    assign_frets(notes, GUITAR.tuning, GUITAR.frets)
    tab = render_ascii_tab(notes, GUITAR.tuning, width=60)
    blocks = tab.split("\n\n")
    assert len(blocks) > 1  # wrapped
    for block in blocks:
        assert len(block.splitlines()) == 6
        assert all(len(ln) <= 60 + 2 for ln in block.splitlines())


# --- guitar wiring -----------------------------------------------------------------------

def test_guitar_config_defaults():
    cfg = GuitarTranscriptionConfig()
    assert len(cfg.tuning) == 6
    assert cfg.tuning[0] == 40 and cfg.tuning[-1] == 64  # E2..E4
    assert cfg.min_frequency < 82.4  # room for drop tunings


def test_pipeline_auto_upgrades_model_for_guitar(tmp_path):
    from drum_extractor.pipeline import run_pipeline

    config = PipelineConfig(
        output_dir=tmp_path,
        do_separation=False,  # nothing to separate; we only check the upgrade
        do_drum_transcription=False,
        do_bass_transcription=False,
        do_guitar_transcription=True,
        do_quantize=False,
        do_notation=False,
        do_sonify=False,
    )
    result = run_pipeline(tmp_path / "song.wav", config)
    assert "guitar" in config.separation.stems
    assert config.separation.model == "htdemucs_6s"
    assert any("guitar" in w for w in result.warnings)  # no stem -> clear warning


def test_cli_guitar_flags_parse():
    from drum_extractor.cli import build_parser

    args = build_parser().parse_args(["run", "s.mp3", "--guitar"])
    assert args.guitar is True
    assert build_parser().parse_args(["run", "s.mp3"]).guitar is False
    tg = build_parser().parse_args(["transcribe-guitar", "g.wav"])
    assert tg.command == "transcribe-guitar"


def test_web_factory_guitar_mode():
    pytest.importorskip("flask")
    from pathlib import Path

    from drum_extractor.web.server import default_config_factory

    make = default_config_factory(Path("out"))
    plain = make("j1")
    assert "guitar" not in plain.separation.stems
    six = make("j2", guitar=True)
    assert six.separation.model == "htdemucs_6s"
    assert "guitar" in six.separation.stems and "piano" in six.separation.stems
    assert six.do_guitar_transcription is True


def test_guitar_transcription_on_synth_chord(tmp_path):
    pytest.importorskip("basic_pitch")
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf

    from drum_extractor.guitar import render_guitar_tab, transcribe_guitar

    sr = 44100
    t = np.arange(2 * sr) / sr
    # E2 + B2 power-chord tones with a little harmonic content.
    sig = sum(
        0.4 * np.sin(2 * np.pi * f * t) + 0.1 * np.sin(2 * np.pi * 2 * f * t)
        for f in (82.41, 123.47)
    ) * np.exp(-((t * 2) % 1) * 2)
    p = tmp_path / "guitar.wav"
    sf.write(str(p), (sig / np.abs(sig).max() * 0.8).astype("float32"), sr)

    notes = transcribe_guitar(p)
    assert notes, "no notes transcribed from a clean synth chord"
    pitches = {n.pitch for n in notes}
    assert any(abs(pc - 40) <= 1 for pc in pitches)  # found ~E2
    tab = render_guitar_tab(notes)
    assert tab.count("\n") >= 5  # six string rows
    assert any(ch.isdigit() for ch in tab)
