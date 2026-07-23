from drum_extractor.config import QuantizeConfig
from drum_extractor.events import DrumHit, Transcription
from drum_extractor.quantize import _annotate_bar_beat, _build_grid, _snap


def test_build_grid_subdivides_beats():
    beats = [0.0, 0.5, 1.0]  # 120 BPM
    grid = _build_grid(beats, QuantizeConfig(grid=16, beat_unit=4))
    # 16th grid over quarter beats = 4 slots per beat.
    assert 0.0 in grid
    assert any(abs(g - 0.125) < 1e-9 for g in grid)  # first sixteenth
    assert any(abs(g - 0.25) < 1e-9 for g in grid)


def test_snap_picks_nearest():
    grid = [0.0, 0.125, 0.25, 0.375, 0.5]
    assert _snap(0.13, grid) == 0.125
    assert _snap(0.24, grid) == 0.25
    assert _snap(0.0, grid) == 0.0


def test_annotate_bar_beat():
    t = Transcription(drum_hits=[DrumHit(time=0.0, instrument="kick"), DrumHit(time=2.1, instrument="snare")])
    bar_starts = [0.0, 2.0, 4.0]  # 2s bars
    _annotate_bar_beat(t, bar_starts, QuantizeConfig(beats_per_bar=4))
    assert t.drum_hits[0].bar == 1
    assert t.drum_hits[0].beat == 0.0
    assert t.drum_hits[1].bar == 2
    assert t.drum_hits[1].beat is not None and t.drum_hits[1].beat >= 0
