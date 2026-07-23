from drum_extractor.bass import assign_tab, render_ascii_tab
from drum_extractor.config import BassTranscriptionConfig
from drum_extractor.events import BassNote


def test_assign_tab_open_strings():
    config = BassTranscriptionConfig()  # E1 A1 D2 G2 = 28 33 38 43
    notes = [BassNote(0.0, 0.5, 28), BassNote(0.5, 1.0, 33), BassNote(1.0, 1.5, 43)]
    assign_tab(notes, config)
    # Low E on the lowest string at fret 0.
    assert (notes[0].string, notes[0].fret) == (0, 0)
    # A can be open on string 1 (fret 0) or fret 5 on string 0; both are valid,
    # assert it is at least playable and on a real string.
    assert notes[1].string is not None and 0 <= notes[1].fret <= config.frets


def test_assign_tab_minimizes_movement():
    config = BassTranscriptionConfig()
    # A2 (45) then B2 (47): should stay on the same string, adjacent frets.
    notes = [BassNote(0.0, 0.25, 45), BassNote(0.25, 0.5, 47)]
    assign_tab(notes, config)
    assert notes[0].string == notes[1].string
    assert abs(notes[0].fret - notes[1].fret) == 2


def test_render_ascii_tab_has_all_strings():
    config = BassTranscriptionConfig()
    notes = [BassNote(0.0, 0.5, 28), BassNote(0.5, 1.0, 40)]
    assign_tab(notes, config)
    tab = render_ascii_tab(notes, config)
    assert tab.count("\n") == 3  # 4 strings -> 4 lines, 3 newlines
    assert "|" in tab
