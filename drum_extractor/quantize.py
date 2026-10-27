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
    if config.backend not in ("madmom", "librosa", "none"):
        raise ValueError(f"Unknown quantize backend: {config.backend!r}")
    if config.grid_mode not in ("tracked", "constant"):
        raise ValueError(f"Unknown grid_mode: {config.grid_mode!r} (use 'tracked' or 'constant')")
    if config.backend == "none":
        return transcription

    tempo, beats, downbeats, bars_tracked = _detect_beats(Path(audio_path), config)
    if not bars_tracked:
        phase = _drum_bar_phase(transcription.drum_hits, beats, config.beats_per_bar)
        if phase:
            downbeats = beats[phase:: config.beats_per_bar]
        log.info(
            "Bar phase %s (librosa tracks beats, not bars; install madmom for tracked downbeats).",
            "read from the drums: kick on 1, snare on 2 and 4" if phase else "assumed from the first beat",
        )

    hit_times = [h.time for h in transcription.drum_hits]
    t_lo = min(hit_times) if hit_times else None
    t_hi = max(hit_times) if hit_times else None

    if config.grid_mode == "constant":
        tempo, beats, downbeats = _constant_beats(tempo, beats, downbeats, config, t_lo, t_hi)

    transcription.tempo = tempo
    transcription.beats = beats
    transcription.downbeats = downbeats
    transcription.time_signature = (config.beats_per_bar, config.beat_unit)

    grid = _build_grid(beats, config, t_lo, t_hi)
    if grid:
        for hit in transcription.drum_hits:
            # Clamp at 0: an extrapolated grid slot just before t=0 must not pull
            # an early onset to a negative time (negative times are dropped by
            # MIDI writers and crash sonification's sample indexing).
            hit.time = max(0.0, _snap(hit.time, grid))
        transcription.drum_hits = _dedupe_hits(transcription.drum_hits)
        _annotate_bar_beat(transcription, downbeats or beats, config)
    log.info("Quantized to 1/%d grid at %.1f BPM (%d beats, %d bars)", config.grid, tempo or 0.0, len(beats), len(downbeats))
    return transcription


def _detect_beats(audio_path: Path, config: QuantizeConfig) -> tuple[float | None, list[float], list[float], bool]:
    """Return ``(tempo, beats, downbeats, bars_tracked)``; the flag is False when
    the downbeats are only assumed from the beats (the librosa backend)."""
    if config.backend == "madmom":
        try:
            return (*_detect_madmom(audio_path, config), True)
        except Exception as exc:  # madmom has heavy/old deps; fall back cleanly
            log.warning("madmom beat tracking failed (%s); using librosa.", exc)
    return (*_detect_librosa(audio_path, config), False)


# How strongly a hit on each beat of the bar says "this phase is right", for the
# rock/punk/metal backbeat: kick on 1 (a little on 3), snare on 2 and 4, crash
# on 1. Hats and ride keep time on every beat, so they abstain.
_PHASE_VOTES: dict[int, dict[str, dict[int, float]]] = {
    4: {"kick": {0: 1.0, 2: 0.25}, "snare": {1: 1.0, 3: 1.0}, "crash": {0: 0.75}},
    2: {"kick": {0: 1.0}, "snare": {1: 1.0}, "crash": {0: 0.75}},
}


def _drum_bar_phase(hits: list, beats: list[float], beats_per_bar: int) -> int:
    """Offset into ``beats`` of the first bar's beat one, read from the drums.

    librosa tracks beats but not bars, and its first tracked beat is often not
    a downbeat (it trims a weak opening beat), so "every Nth beat from the
    first" can displace the whole chart — the backbeat engraved on 1 and 3.
    Each candidate phase is scored by the votes above. A phase and its
    half-bar twin share all the backbeat evidence (only kick-on-1 tells them
    apart), so the winner must clearly beat every OTHER phase; choosing between
    twins can only move the chart by half a bar, never off the backbeat.
    Returns 0 (keep the naive phase) when the meter has no vote table or the
    drums don't clearly prefer a phase: blast beats, half-time, sparse intros.
    """
    votes = _PHASE_VOTES.get(beats_per_bar)
    if not votes or len(beats) < 2 * beats_per_bar or not hits:
        return 0
    import bisect
    import statistics

    from .gm_drum_map import CRASH, FAMILY

    period = statistics.median(b - a for a, b in zip(beats, beats[1:]))
    tol = 0.15 * period  # well inside a 16th, so off-beat strokes don't vote
    scores = [0.0] * beats_per_bar
    for h in hits:
        table = votes.get("crash" if h.instrument == CRASH else FAMILY.get(h.instrument, ""))
        if not table:
            continue
        i = bisect.bisect_left(beats, h.time)
        near = min((j for j in (i - 1, i) if 0 <= j < len(beats)), key=lambda j: abs(beats[j] - h.time))
        if abs(beats[near] - h.time) > tol:
            continue
        for phase in range(beats_per_bar):
            scores[phase] += table.get((near - phase) % beats_per_bar, 0.0)

    best = max(range(beats_per_bar), key=scores.__getitem__)  # ties keep phase 0
    twin = (best + beats_per_bar // 2) % beats_per_bar
    rivals = [s for p, s in enumerate(scores) if p != best and (p != twin or p == 0)]
    # A couple of bars of evidence and a clear margin, or leave the phase alone.
    if best == 0 or scores[best] < 4.0 or scores[best] < 1.2 * max(rivals):
        return 0
    return best


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
    else:
        # madmom's default 55-215 BPM window octave-errors on fast material;
        # let callers widen/narrow the search (e.g. a metal preset).
        if config.min_bpm:
            dbn_kwargs["min_bpm"] = config.min_bpm
        if config.max_bpm:
            dbn_kwargs["max_bpm"] = config.max_bpm
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
    elif config.min_bpm and config.max_bpm:
        # librosa has no hard min/max; steer its prior toward the range midpoint.
        bt_kwargs["start_bpm"] = (config.min_bpm + config.max_bpm) / 2.0
    tempo, beat_frames = librosa.beat.beat_track(**bt_kwargs)
    beats = [float(t) for t in librosa.frames_to_time(beat_frames, sr=sr)]
    # librosa >=0.10 returns tempo as a NumPy array; extract a scalar safely.
    tempo_arr = np.atleast_1d(tempo)
    tempo_scalar = float(tempo_arr.flat[0]) if tempo_arr.size and tempo_arr.flat[0] > 0 else None
    tempo_val = config.fixed_tempo or tempo_scalar or _tempo_from_beats(beats)
    # librosa gives beats but not downbeats; assume bar starts every N beats.
    # That bar PHASE is a guess, which quantize() corrects from the drums.
    downbeats = beats[:: config.beats_per_bar] if beats else []
    return tempo_val, beats, downbeats


def _constant_beats(
    tempo: float | None,
    beats: list[float],
    downbeats: list[float],
    config: QuantizeConfig,
    t_lo: float | None,
    t_hi: float | None,
) -> tuple[float | None, list[float], list[float]]:
    """Replace tracked beats with a uniform grid at a constant tempo.

    note-seq-style quantization: one steps-per-second rate anchored at the first
    detected downbeat. For steady-tempo songs this is more robust than the
    tracked grid, where a single missed beat halves the local resolution.
    Falls back to the tracked beats when no tempo is available.
    """
    tempo_c = config.fixed_tempo or tempo
    if not tempo_c or tempo_c <= 0:
        log.warning("grid_mode='constant' needs a tempo (fixed or detected); using tracked beats.")
        return tempo, beats, downbeats

    import math

    period = 60.0 / tempo_c
    bpb = config.beats_per_bar
    anchor = downbeats[0] if downbeats else (beats[0] if beats else 0.0)
    lo = t_lo if t_lo is not None else anchor
    hi = t_hi if t_hi is not None else anchor + bpb * period

    # Extend backwards in WHOLE BARS so the anchor stays a downbeat — bounded
    # like every other extension guard, so one absurd stray time can't starve
    # the forward beat count (which shares the 100k cap). Only extend when an
    # onset is more than HALF A GRID STEP early: onset detectors routinely fire
    # a few ms before the tracked downbeat, and those hits snap forward onto
    # the anchor anyway — extending for them would prepend a bar that ends up
    # completely empty (shifting every bar number and creating negative beats).
    half_step = period / max(1, config.grid // config.beat_unit) / 2.0
    n_back = max(0, math.ceil((anchor - lo - half_step) / period)) if lo < anchor - half_step else 0
    n_back = min(n_back, 256 * bpb)
    n_back += (-n_back) % bpb
    start = anchor - n_back * period
    count = min(int((hi - start) / period) + 2, 100_000)
    beats_c = [start + i * period for i in range(count)]
    downbeats_c = beats_c[::bpb]
    log.info("Constant grid: %.1f BPM anchored at %.3fs (%d beats)", tempo_c, anchor, count)
    return tempo_c, beats_c, downbeats_c


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


def _build_grid(
    beats: list[float],
    config: QuantizeConfig,
    t_lo: float | None = None,
    t_hi: float | None = None,
) -> list[float]:
    """Subdivide each beat interval into ``grid / beat_unit`` slots.

    Two robustness measures over the naive version:
    - An interval much longer than the median beat (a missed beat) is treated as
      multiple beats and subdivided accordingly, so one tracking dropout doesn't
      halve the local grid resolution.
    - The grid is extrapolated (at the median beat period) to cover ``t_lo`` /
      ``t_hi``, so onsets before the first or after the last detected beat snap
      to a real slot instead of being clamped onto the endpoint.
    """
    if len(beats) < 2:
        return []
    import statistics

    subdivisions = max(1, config.grid // config.beat_unit)
    intervals = [b - a for a, b in zip(beats, beats[1:]) if b > a]
    if not intervals:
        return []
    median_iv = statistics.median(intervals)

    # Split oversized intervals into the number of beats they most likely span.
    expanded = [beats[0]]
    for a, b in zip(beats, beats[1:]):
        gap = b - a
        if gap <= 0:
            continue
        k = max(1, round(gap / median_iv)) if gap > 1.6 * median_iv else 1
        step_b = gap / k
        expanded.extend(a + i * step_b for i in range(1, k + 1))

    # Extrapolate to cover the onset range (bounded so a stray time can't explode it).
    guard = 0
    while t_lo is not None and expanded[0] > t_lo + 1e-9 and guard < 512:
        expanded.insert(0, expanded[0] - median_iv)
        guard += 1
    guard = 0
    while t_hi is not None and expanded[-1] < t_hi - 1e-9 and guard < 512:
        expanded.append(expanded[-1] + median_iv)
        guard += 1

    grid: list[float] = []
    for a, b in zip(expanded, expanded[1:]):
        step = (b - a) / subdivisions
        grid.extend(a + i * step for i in range(subdivisions))
    grid.append(expanded[-1])
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
    # land in a real earlier bar with a non-negative beat, and append synthetic
    # bars past the last downbeat so an outro doesn't pile every hit onto the
    # final detected bar's end (both capped so a stray onset can't explode the
    # bar count).
    starts = list(bar_starts)
    earliest = min((h.time for h in transcription.drum_hits), default=starts[0])
    guard = 0
    while earliest < starts[0] - 1e-9 and guard < 64:
        starts.insert(0, starts[0] - default_span)
        guard += 1
    latest = max((h.time for h in transcription.drum_hits), default=starts[-1])
    guard = 0
    while latest > starts[-1] + default_span - 1e-9 and guard < 256:
        starts.append(starts[-1] + default_span)
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
