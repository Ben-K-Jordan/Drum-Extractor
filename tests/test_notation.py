"""Notation tests. Skipped automatically if music21 is not installed."""

import pytest

pytest.importorskip("music21")

from drum_extractor.config import NotationConfig, QuantizeConfig  # noqa: E402
from drum_extractor.events import DrumHit, Transcription  # noqa: E402
from drum_extractor.notation import build_score, transcription_to_musicxml  # noqa: E402


def _basic_rock_beat() -> Transcription:
    """One bar of a simple rock groove at 120 BPM (kick/snare/hi-hat)."""
    hits = []
    for i in range(8):  # eighth-note hi-hats
        hits.append(DrumHit(time=i * 0.25, instrument="hihat_closed", bar=1, beat=i * 0.5))
    hits.append(DrumHit(time=0.0, instrument="kick", bar=1, beat=0.0))
    hits.append(DrumHit(time=1.0, instrument="kick", bar=1, beat=2.0))
    hits.append(DrumHit(time=0.5, instrument="snare", bar=1, beat=1.0))
    hits.append(DrumHit(time=1.5, instrument="snare", bar=1, beat=3.0))
    return Transcription(drum_hits=hits, tempo=120.0, time_signature=(4, 4))


def test_build_score_produces_notes():
    score = build_score(_basic_rock_beat(), NotationConfig(), QuantizeConfig(grid=16))
    notes = list(score.recurse().notes)
    assert len(notes) > 0


def test_musicxml_export(tmp_path):
    out = transcription_to_musicxml(_basic_rock_beat(), tmp_path / "drums.musicxml", NotationConfig(), QuantizeConfig())
    assert out.exists()
    content = out.read_text()
    assert "score-partwise" in content
    # Percussion clef should be present in the exported staff.
    assert "percussion" in content.lower() or "clef" in content.lower()
