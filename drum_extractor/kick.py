"""Phase 4 — double-kick / double-bass onset booster.

The single biggest weakness of drum transcription on metal is undercounted fast
double-kick: full-kit models merge closely-spaced kick transients. This module
isolates the kick band with a low-pass filter and runs a dedicated onset
detector tuned for tight spacing, then merges the recovered kicks into the main
transcription.

It's a pragmatic DSP approach that runs with just scipy+librosa. A heavier
LarsNet kick stem can be plugged in via ``kick_stem`` if you have it, but the
low-pass path works today and needs no extra model.
"""

from __future__ import annotations

from pathlib import Path

from .errors import MissingDependencyError
from .events import DrumHit
from .gm_drum_map import KICK
from .logging_utils import get_logger

log = get_logger(__name__)


def detect_kick_onsets(
    drum_stem: str | Path,
    cutoff_hz: float = 140.0,
    min_gap_ms: float = 30.0,
    sr: int = 44100,
) -> list[float]:
    """Return kick onset times (seconds) from an isolated low-frequency band.

    ``min_gap_ms`` sets the tightest spacing resolved; 30 ms ~= 500 BPM sixteenth
    double-kick, so this can resolve blast-tempo feet the full-kit model blurs.
    """
    try:
        import librosa  # type: ignore
        import numpy as np  # type: ignore
        from scipy.signal import butter, sosfiltfilt  # type: ignore
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Double-kick booster", "librosa", extra="drums") from exc

    y, sr = librosa.load(str(drum_stem), sr=sr, mono=True)
    if not np.isfinite(y).all():
        log.warning("Drum stem contains non-finite samples; sanitizing for the kick booster.")
        y = np.nan_to_num(y)
    # butter() requires cutoff < Nyquist; clamp so a small analysis sr can't
    # crash filter design with an opaque scipy error.
    cutoff = min(cutoff_hz, 0.45 * sr)
    sos = butter(4, cutoff, btype="low", fs=sr, output="sos")
    # sosfiltfilt pads by ~3*(2*n_sections+1) samples and raises on shorter
    # input; guard so a truncated/near-empty stem can't crash transcription.
    padlen = 3 * (2 * sos.shape[0] + 1)
    if y.size <= padlen:
        log.warning("Drum stem too short (%d samples) for the kick booster; skipping.", int(y.size))
        return []
    y_low = sosfiltfilt(sos, y).astype(np.float32)

    hop = 128  # ~2.9 ms at 44.1k -> fine enough for fast feet
    # n_fft=512: the default 2048 (46 ms) window makes every kick's envelope
    # fire TWICE ~50 ms apart, fabricating phantom 32nd-note double-kicks on
    # ordinary grooves. The short window still resolves 45 ms-gap real doubles.
    env = librosa.onset.onset_strength(y=y_low, sr=sr, hop_length=hop, n_fft=512)
    wait = max(1, int((min_gap_ms / 1000.0) * sr / hop))
    # Peaks first WITHOUT backtracking: the echo filter below needs each
    # peak's envelope value, and backtracking rewinds to the local minimum.
    peaks = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop, units="frames", backtrack=False,
        wait=wait, pre_max=wait, post_max=wait,
    )
    if len(peaks) == 0:
        return []
    # Echo filter: a kick's decaying body can raise a secondary envelope bump
    # 50-100 ms after the true attack — the same spacing as real fast
    # double-kick, so time alone can't separate them. Magnitude can: real
    # consecutive strokes have comparable envelope peaks, while the echo is a
    # small fraction of its parent's.
    echo_window = int(0.25 * sr / hop)
    kept: list[int] = []
    for p in peaks:
        p = int(p)
        recent = [env[q] for q in kept if p - q <= echo_window]
        if recent and env[p] < 0.5 * max(recent):
            continue
        kept.append(p)
    frames = librosa.onset.onset_backtrack(np.asarray(kept, dtype=int), env)
    onsets = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    # Belt-and-braces: enforce the min gap on the final times (backtracking can
    # land two peaks closer together than the peak-picker's `wait` spacing).
    min_gap = min_gap_ms / 1000.0
    times: list[float] = []
    for t in onsets:
        if not times or t - times[-1] >= min_gap:
            times.append(float(t))
    return times


def boost_double_kick(
    hits: list[DrumHit],
    drum_stem: str | Path,
    velocity: int = 96,
    merge_window_ms: float = 40.0,
    cutoff_hz: float = 140.0,
    min_gap_ms: float = 30.0,
) -> list[DrumHit]:
    """Augment ``hits`` with kicks recovered from the low-frequency band.

    A detected kick onset is added only if no existing kick sits within
    ``merge_window_ms`` — so we fill in missed fast kicks without doubling ones
    the main transcriber already found.
    """
    kick_times = detect_kick_onsets(drum_stem, cutoff_hz=cutoff_hz, min_gap_ms=min_gap_ms)
    # Snapshot the main-transcription kicks. Recovered kicks are de-duplicated
    # ONLY against this fixed snapshot, never against each other, so genuinely
    # distinct fast kicks (already spaced >= min_gap_ms by the detector) all
    # survive. Previously we insort-ed each added kick, so a recovered kick
    # would fall inside the next one's merge window and get dropped every-other.
    original = sorted(h.time for h in hits if h.instrument == KICK)
    window = merge_window_ms / 1000.0

    import bisect

    added = 0
    out = list(hits)
    for t in kick_times:
        idx = bisect.bisect_left(original, t)
        near = any(0 <= j < len(original) and abs(original[j] - t) <= window for j in (idx - 1, idx))
        if not near:
            out.append(DrumHit(time=t, instrument=KICK, velocity=velocity))
            added += 1
    out.sort(key=lambda h: h.time)
    log.info("Double-kick booster: %d kick onsets detected, %d new kicks added", len(kick_times), added)
    return out
