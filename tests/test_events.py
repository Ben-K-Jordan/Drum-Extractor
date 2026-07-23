from drum_extractor.events import BassNote, DrumHit, Transcription


def test_transcription_roundtrip():
    t = Transcription(
        drum_hits=[DrumHit(time=0.0, instrument="kick", velocity=100, bar=1, beat=0.0)],
        bass_notes=[BassNote(start=0.0, end=0.5, pitch=28, velocity=90, string=0, fret=0)],
        tempo=180.0,
        beats=[0.0, 0.33, 0.66],
        downbeats=[0.0],
        time_signature=(4, 4),
    )
    restored = Transcription.from_dict(t.to_dict())
    assert restored.tempo == 180.0
    assert restored.time_signature == (4, 4)
    assert restored.drum_hits[0].instrument == "kick"
    assert restored.bass_notes[0].pitch == 28
    assert restored.drum_hits[0].bar == 1


def test_empty_transcription_serializes():
    t = Transcription()
    d = t.to_dict()
    assert d["drum_hits"] == []
    assert d["time_signature"] == [4, 4]
