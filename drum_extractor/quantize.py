"""Stage 3 — beat tracking and quantization.

Snaps raw onset times onto a musical grid so the notation stage can place notes
in bars. Readability of the final sheet depends heavily on this step: too coarse
a grid drops fast fills, too fine explodes into 32nd-note clutter. Default is a
1/16 grid; use 1/32 for fast double-kick metal.

Primary backend is madmom (strong beat/downbeat/tempo tracking); a librosa
fallback keeps the stage working without it.
"""

from __future__ import annotations

from pathlib import Path

from .config import QuantizeConfig
from .events import Transcription
from .logging_utils import get_logger

log = get_logger(__name__)


def quantize(transcription: Transcription, audio_path: str | Path, config: QuantizeConfig | None = None) -> Transcription:
    """Detect tempo/beats from ``audio_path`` and snap onsets to the grid.

    Mutates and returns ``transcription`` with ``tempo``, ``beats``,
    ``downbeats`` filled in and each hit's ``bar``/``beat`` annotated.
    """
    config = config or QuantizeConfig()
    if config.backend == "none":
        return transcription

    tempo, beats, downbeats = _detect_beats(Path(audio_path), config)
    transcription.tempo = tempo
    transcription.beats = beats
    transcription.downbeats = downbeats
    transcription.time_signature = (config.beats_per_bar, config.beat_unit)

    grid = _build_grid(beats, config)
    if grid:
        for hit in transcription.drum_hits:
            hit.time = _snap(hit.time, grid)
        transcription.drum_hits = _dedupe_hits(transcription.drum_hits)
        _annotate_bar_beat(transcription, downbeats or beats, config)
    log.info("Quantized to 1/%d grid at %.1f BPM (%d beats, %d bars)", config.grid, tempo or 0.0, len(beats), len(downbeats))
    return transcription


def _detect_beats(audio_path: Path, config: QuantizeConfig) -> tuple[float | None, list[float], list[float]]:
    if config.backend == "madmom":
        try:
            return _detect_madmom(audio_path, config)
        except Exception as exc:  # madmom has heavy/old deps; fall back cleanly
            log.warning("madmom beat tracking failed (%s); using librosa.", exc)
    return _detect_librosa(audio_path, config)


def _detect_madmom(audio_path: Path, config: QuantizeConfig) -> tuple[float | None, list[float], list[float]]:
    try:
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor  # type: ignore
    except ModuleNotFoundError as exc:
        from .errors import MissingDependencyError

        raise MissingDependencyError("Beat tracking", "madmom", extra="quantize") from exc

    act = RNNDownBeatProcessor()(str(audio_path))
    dbn_kwargs = {"beats_per_bar": [config.beats_per_bar], "fps": 100}
    if config.fixed_tempo:
        # Constrain tracking to a tight window around the known tempo instead of
        # only relabeling the reported value afterwards.
        dbn_kwargs["min_bpm"] = config.fixed_tempo * 0.97
        dbn_kwargs["max_bpm"] = config.fixed_tempo * 1.03
    proc = DBNDownBeatTrackingProcessor(**dbn_kwargs)
    result = proc(act)  # rows of [time, beat_number]
    beats = [float(t) for t, _ in result]
    downbeats = [float(t) for t, b in result if int(b) == 1]
    tempo = config.fixed_tempo or _tempo_from_beats(beats)
    return tempo, beats, downbeats


def _detect_librosa(audio_path: Path, config: QuantizeConfig) -> tuple[float | None, list[float], list[float]]:
    try:
        import librosa  # type: ignore
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        from .errors import MissingDependencyError

        raise MissingDependencyError("Beat tracking", "librosa", extra="drums") from exc

    y, sr = librosa.load(str(audio_path), sr=44100, mono=True)
    bt_kwargs = {"y": y, "sr": sr, "units": "frames"}
    if config.fixed_tempo:
        # bpm= fixes the tempo used for the beat-tracking DP (not just a prior).
        bt_kwargs["bpm"] = float(config.fixed_tempo)
    tempo, beat_frames = librosa.beat.beat_track(**bt_kwargs)
    beats = [float(t) for t in librosa.frames_to_time(beat_frames, sr=sr)]
    # librosa >=0.10 returns tempo as a NumPy array; extract a scalar safely.
    tempo_arr = np.atleast_1d(tempo)
    tempo_scalar = float(tempo_arr.flat[0]) if tempo_arr.size and tempo_arr.flat[0] > 0 else None
    tempo_val = config.fixed_tempo or tempo_scalar or _tempo_from_beats(beats)
    # librosa gives beats but not downbeats; assume bar starts every N beats.
    downbeats = beats[:: config.beats_per_bar] if beats else []
    return tempo_val, beats, downbeats


def _dedupe_hits(hits: list) -> list:
    """Collapse hits with the same instrument at the same (snapped) time.

    After quantization two nearby onsets can land on the same grid slot; without
    this they'd produce doubled noteheads and doubled MIDI notes. Keeps the
    loudest of each duplicate group.
    """
    best: dict[tuple[str, float], object] = {}
    for h in hits:
        key = (h.instrument, round(h.time, 4))
        if key not in best or h.velocity > best[key].velocity:  # type: ignore[attr-defined]
            best[key] = h
    return sorted(best.values(), key=lambda h: (h.time, h.instrument))  # type: ignore[attr-defined]


def _tempo_from_beats(beats: list[float]) -> float | None:
    if len(beats) < 2:
        return None
    import statistics

    intervals = [b - a for a, b in zip(beats, beats[1:]) if b > a]
    if not intervals:
        return None
    return 60.0 / statistics.median(intervals)


def _build_grid(beats: list[float], config: QuantizeConfig) -> list[float]:
    """Subdivide each beat interval into ``grid / beat_unit`` slots."""
    if len(beats) < 2:
        return []
    subdivisions = max(1, config.grid // config.beat_unit)
    grid: list[float] = []
    for a, b in zip(beats, beats[1:]):
        step = (b - a) / subdivisions
        grid.extend(a + i * step for i in range(subdivisions))
    grid.append(beats[-1])
    return grid


def _snap(t: float, grid: list[float]) -> float:
    import bisect

    idx = bisect.bisect_left(grid, t)
    candidates = []
    if idx < len(grid):
        candidates.append(grid[idx])
    if idx > 0:
        candidates.append(grid[idx - 1])
    return min(candidates, key=lambda g: abs(g - t)) if candidates else t


def _annotate_bar_beat(transcription: Transcription, bar_starts: list[float], config: QuantizeConfig) -> None:
    """Fill in 1-indexed bar number and beat-within-bar (in beat units).

    Robust to (a) hits before the first detected downbeat — a pickup/anacrusis,
    common with the madmom backend where ``downbeats[0]`` can be later than
    ``beats[0]`` — which would otherwise produce a negative beat and crash the
    notation stage; and (b) the final bar, whose length is estimated from the
    surrounding bars / tempo rather than a hardcoded 2.0s (only correct at
    120 BPM 4/4).
    """
    if not bar_starts:
        return
    import bisect

    # Representative bar length: median of detected inter-downbeat gaps, else
    # derived from tempo, else a last-resort constant.
    spans = sorted(b - a for a, b in zip(bar_starts, bar_starts[1:]) if b > a)
    if spans:
        default_span = spans[len(spans) // 2]
    elif transcription.tempo:
        default_span = config.beats_per_bar * 60.0 / transcription.tempo
    else:
        default_span = 2.0

    # Prepend synthetic bar starts so any pickup hits before the first downbeat
    # land in a real earlier bar with a non-negative beat (capped so a wildly
    # early stray onset can't explode the bar count).
    starts = list(bar_starts)
    earliest = min((h.time for h in transcription.drum_hits), default=starts[0])
    guard = 0
    while earliest < starts[0] - 1e-9 and guard < 64:
        starts.insert(0, starts[0] - default_span)
        guard += 1

    bpb = config.beats_per_bar
    for hit in transcription.drum_hits:
        bar_idx = bisect.bisect_right(starts, hit.time) - 1
        if bar_idx < 0:
            bar_idx = 0
        hit.bar = bar_idx + 1
        bar_start = starts[bar_idx]
        bar_end = starts[bar_idx + 1] if bar_idx + 1 < len(starts) else bar_start + default_span
        span = max(bar_end - bar_start, 1e-6)
        beat = (hit.time - bar_start) / span * bpb
        hit.beat = round(min(max(beat, 0.0), bpb), 4)
