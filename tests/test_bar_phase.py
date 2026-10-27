"""Bar phase from the drums when the beat tracker can't find downbeats."""

from __future__ import annotations

import pytest

from drum_extractor.config import QuantizeConfig
from drum_extractor.events import DrumHit, Transcription
from drum_extractor.quantize import _drum_bar_phase

BEAT = 0.5


def _backbeat_groove(bars: int = 6) -> list[DrumHit]:
    """Kick pair on 1 and the '&' of 1, kicks on the 'a' of 2 and '&' of 3,
    snare on 2 and 4, straight-8th hats — the demo's groove."""
    hits = []
    for bar in range(bars):
        b0 = bar * 4 * BEAT
        hits += [DrumHit(b0 + e * BEAT / 2, "hihat_closed", 90) for e in range(8)]
        hits += [DrumHit(b0 + beat * BEAT, "kick", 110) for beat in (0.0, 0.5, 1.75, 2.5)]
        hits += [DrumHit(b0 + beat * BEAT, "snare", 115) for beat in (1.0, 3.0)]
    return hits


def test_phase_found_when_first_tracked_beat_is_the_snare():
    # librosa trimmed the opening downbeat: tracking starts on beat 2.
    beats = [BEAT * k for k in range(1, 24)]
    phase = _drum_bar_phase(_backbeat_groove(), beats, 4)
    assert phase == 3
    assert beats[phase] == pytest.approx(4 * BEAT)  # the second bar's one


def test_correct_naive_phase_is_kept():
    beats = [BEAT * k for k in range(24)]
    assert _drum_bar_phase(_backbeat_groove(), beats, 4) == 0


def test_sparse_snares_and_stray_kicks_still_find_the_backbeat():
    # A rough transcriber: most snares missed, phantom kicks on 3. The phase
    # and its half-bar twin score close, but both crush the naive guess.
    beats = [BEAT * k for k in range(1, 32)]
    hits = [DrumHit(bar * 4 * BEAT, "kick", 110) for bar in range(8)]
    hits += [DrumHit((bar * 4 + 2) * BEAT, "kick", 90) for bar in range(6)]
    hits += [DrumHit((bar * 4 + 1) * BEAT, "snare", 110) for bar in (1, 4, 6)]
    hits += [DrumHit((bar * 4 + 3) * BEAT, "snare", 110) for bar in (0, 2, 3, 5)]
    assert _drum_bar_phase(hits, beats, 4) == 3


def test_four_on_the_floor_lands_the_backbeat_on_two_and_four():
    # Kick on every beat can't say which beat is one, but the snare still
    # rules out the naive phase that puts it on 1 and 3.
    beats = [BEAT * k for k in range(1, 24)]
    hits = [DrumHit(k * BEAT, "kick", 110) for k in range(24)]
    hits += [DrumHit(k * BEAT, "snare", 110) for k in range(1, 24, 2)]
    phase = _drum_bar_phase(hits, beats, 4)
    snare_positions = {(round(h.time / BEAT) - 1 - phase) % 4 for h in hits if h.instrument == "snare"}
    assert snare_positions == {1, 3}


@pytest.mark.parametrize("label, hits", [
    ("blast beat", [DrumHit(k * BEAT / 2, "kick" if k % 2 else "snare", 110) for k in range(48)]),
    ("half-time", [DrumHit(k * 4 * BEAT, "kick", 110) for k in range(6)]
     + [DrumHit((k * 4 + 2) * BEAT, "snare", 110) for k in range(6)]),
    ("hats only", [DrumHit(k * BEAT / 2, "hihat_closed", 90) for k in range(48)]),
])
def test_ambiguous_grooves_keep_the_naive_phase(label, hits):
    beats = [BEAT * k for k in range(1, 24)]
    assert _drum_bar_phase(hits, beats, 4) == 0, label


def test_meters_without_a_backbeat_rule_are_left_alone():
    beats = [BEAT * k for k in range(1, 24)]
    assert _drum_bar_phase(_backbeat_groove(), beats, 3) == 0


def test_quantize_engraves_the_backbeat_on_two_and_four(monkeypatch, tmp_path):
    from drum_extractor import quantize as Q

    beats = [BEAT * k for k in range(1, 24)]
    monkeypatch.setattr(Q, "_detect_librosa", lambda path, cfg: (120.0, beats, beats[::4]))
    tr = Transcription(drum_hits=_backbeat_groove())
    Q.quantize(tr, tmp_path / "unused.wav", QuantizeConfig(backend="librosa", fixed_tempo=120.0, grid_mode="constant"))

    assert tr.downbeats[0] == pytest.approx(0.0)
    snare_beats = {h.beat for h in tr.drum_hits if h.instrument == "snare"}
    assert snare_beats == {1.0, 3.0}  # 0-based: the 2 and the 4
    assert all(h.bar >= 1 for h in tr.drum_hits)


def test_song_starting_a_whole_bar_before_the_anchor_gets_no_empty_bar():
    from drum_extractor.quantize import _constant_beats

    # The tracked downbeat lands a few ms late, exactly one bar after the
    # first hit: the grid must start on that hit, not a whole empty bar early.
    period = 60.0 / 117.0
    anchor = 4 * period + 0.016
    _tempo, beats, downbeats = _constant_beats(
        117.0, [anchor], [anchor], QuantizeConfig(fixed_tempo=117.0, grid_mode="constant"),
        t_lo=0.0, t_hi=8.0,
    )
    assert downbeats[0] == pytest.approx(0.016)


def test_tracked_downbeats_are_never_overridden(monkeypatch, tmp_path):
    from drum_extractor import quantize as Q

    beats = [BEAT * k for k in range(1, 24)]
    tracked = beats[::4]
    monkeypatch.setattr(Q, "_detect_madmom", lambda path, cfg: (120.0, beats, tracked))
    tr = Transcription(drum_hits=_backbeat_groove())
    Q.quantize(tr, tmp_path / "unused.wav", QuantizeConfig(backend="madmom", fixed_tempo=120.0))
    assert tr.downbeats == tracked
