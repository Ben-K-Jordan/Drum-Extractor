"""Tab and Guitar Pro exports read like a part, not a grid dump."""

from __future__ import annotations

import pytest

from drum_extractor.events import BassNote

BASS = (28, 33, 38, 43)
EIGHTH = 0.25  # at 120 BPM


def test_gp5_chugs_are_eighth_notes_not_sixteenths_and_rests(tmp_path):
    gp = pytest.importorskip("guitarpro")
    from drum_extractor.gp_export import write_gp5

    # The demo riff's half-bar: F on eighths 0, 1 and 3, each ringing ~1.8 eighths.
    notes = [BassNote(e * EIGHTH, e * EIGHTH + 1.8 * EIGHTH, 29, string=0, fret=1) for e in (0, 1, 3)]
    song = gp.parse(str(write_gp5(notes, BASS, tmp_path / "c.gp5", tempo=120.0)))
    beats = song.tracks[0].measures[0].voices[0].beats
    first_half = [(b.duration.value, bool(b.notes)) for b in beats[:4]]
    assert first_half == [(8, True), (8, True), (8, False), (8, True)]
    # Every bar still adds up to 4/4 (in 16ths).
    for m in song.tracks[0].measures:
        sixteenths = sum(16 // b.duration.value * (1.5 if b.duration.isDotted else 1) for b in m.voices[0].beats)
        assert sixteenths == 16


def test_gp5_notes_never_cross_a_beat_line(tmp_path):
    gp = pytest.importorskip("guitarpro")
    from drum_extractor.gp_export import write_gp5

    # A long note on the 'a' of 1: written as a 16th, not across beat 2.
    song = gp.parse(str(write_gp5([BassNote(0.375, 1.5, 33, string=1, fret=0)], BASS, tmp_path / "l.gp5",
                                  tempo=120.0)))
    pos = 0
    for b in song.tracks[0].measures[0].voices[0].beats:
        length = 16 // b.duration.value * (1.5 if b.duration.isDotted else 1)
        if b.notes:
            assert pos // 4 == (pos + length - 1) // 4
        pos += length


def test_tab_legend_only_when_the_tab_draws_an_x():
    from drum_extractor.tabs import assign_frets, render_ascii_tab

    fretted = [BassNote(0.0, 0.5, 33)]
    assign_frets(fretted, BASS, 24)
    assert "x = note" not in render_ascii_tab(fretted, BASS, title="t")
    unplaceable = [BassNote(0.0, 0.5, 20)]  # below the low E string
    assign_frets(unplaceable, BASS, 24)
    assert "x = note" in render_ascii_tab(unplaceable, BASS, title="t")
