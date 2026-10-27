"""Drum-chart engraving conventions: visible beats, two voices, clean accents."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from drum_extractor.config import NotationConfig, QuantizeConfig
from drum_extractor.events import DrumHit, Transcription

pytest.importorskip("music21")


def _xml(tmp_path, hits):
    from drum_extractor.notation import transcription_to_musicxml

    path = transcription_to_musicxml(Transcription(drum_hits=hits, tempo=120.0), tmp_path / "c.musicxml",
                                     NotationConfig(), QuantizeConfig())
    return ET.parse(path).getroot()


def _voice_spans(root, measure, voice: str):
    """(start, end, is_rest) in quarter lengths of every note and rest in a voice."""
    divisions = float(next(root.iter("divisions")).text)  # declared once, inherited
    spans, pos = [], 0.0
    for el in measure:
        if el.tag == "backup":
            pos -= float(el.findtext("duration")) / divisions
        elif el.tag == "forward":
            pos += float(el.findtext("duration")) / divisions
        elif el.tag == "note":
            d = float(el.findtext("duration")) / divisions
            if el.find("chord") is not None:
                continue
            if el.findtext("voice") == voice:
                spans.append((pos, pos + d, el.find("rest") is not None))
            pos += d
    return spans


def test_off_beat_kicks_never_hide_a_beat(tmp_path):
    # The demo groove's feet: kick on 1, the '&' of 1, the 'a' of 2, the '&' of 3.
    hits = [DrumHit(0.0, "kick", 110, bar=1, beat=b) for b in (0.0, 0.5, 1.75, 2.5)]
    root = _xml(tmp_path, hits)
    measure = next(root.iter("measure"))
    notes = [(a, b) for a, b, rest in _voice_spans(root, measure, "2") if not rest]
    assert [a for a, _ in notes] == [0.0, 0.5, 1.75, 2.5]
    for start, end in notes:
        assert int(start + 1e-9) == int(end - 1e-9), f"note {start}-{end} crosses a beat line"


def test_bar_with_only_hands_keeps_a_feet_voice_rest(tmp_path):
    hits = [DrumHit(0.0, "kick", 100, bar=1, beat=0.0)]
    hits += [DrumHit(0.0, "hihat_closed", 90, bar=2, beat=b / 2) for b in range(8)]
    root = _xml(tmp_path, hits)
    bar2 = [m for m in root.iter("measure") if m.get("number") == "2"][0]
    voices = {n.findtext("voice") for n in bar2.iter("note")}
    assert voices == {"1", "2"}
    assert any(rest for _, _, rest in _voice_spans(root, bar2, "2"))


def test_accents_are_single_and_above_the_staff(tmp_path):
    # An accented snare under an accented hat: one '>' on the chord, above.
    hits = [DrumHit(0.0, "hihat_closed", 120, bar=1, beat=1.0), DrumHit(0.0, "snare", 125, bar=1, beat=1.0)]
    root = _xml(tmp_path, hits)
    accents = list(root.iter("accent"))
    assert len(accents) == 1
    assert accents[0].get("placement") == "above"


def test_rests_never_hide_the_middle_of_the_bar(tmp_path):
    # Kick on 1, the '&' of 1 and the '&' of 3: the gap from beat 2 to the
    # '&' of 3 must not be one dotted-quarter rest straddling beat 3.
    hits = [DrumHit(0.0, "kick", 110, bar=1, beat=b) for b in (0.0, 0.5, 2.5)]
    root = _xml(tmp_path, hits)
    measure = next(root.iter("measure"))
    rests = [(a, b) for a, b, rest in _voice_spans(root, measure, "2") if rest]
    assert rests, "expected rests in the feet voice"
    for start, end in rests:
        assert not (start < 2.0 - 1e-9 < end - 1e-9), f"rest {start}-{end} hides beat 3"
        if start % 1 > 1e-9:
            assert end <= int(start) + 1 + 1e-9, f"off-beat rest {start}-{end} crosses a beat line"
