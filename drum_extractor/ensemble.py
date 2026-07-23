"""Phase 4 — ensemble separation to reduce distorted-guitar bleed in drums.

The standard community fix for dense metal is to run a second, different drum
separator and average it with Demucs: uncorrelated bleed partially cancels while
the drums reinforce. This module provides the (testable) waveform-averaging core
plus a pluggable hook to produce the second drum stem via ``audio-separator``
(python-audio-separator), which exposes RoFormer / SCNet / MDX drum models.

Averaging is pure numpy+soundfile. The second model is optional — if it isn't
available the pipeline simply uses the Demucs drums unchanged.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .errors import ExternalToolError, MissingDependencyError
from .logging_utils import get_logger

log = get_logger(__name__)


def _best_lag(a, b, max_lag: int):
    """Integer-sample lag that best aligns mono signal ``b`` to ``a`` (|lag|<=max_lag)."""
    import numpy as np  # type: ignore
    from scipy.signal import correlate  # type: ignore

    n = min(len(a), len(b))
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    if not np.any(a) or not np.any(b):
        return 0
    corr = correlate(a, b, mode="full", method="fft")
    center = n - 1
    lo, hi = max(0, center - max_lag), min(len(corr), center + max_lag + 1)
    return int((np.arange(lo, hi) - center)[np.argmax(corr[lo:hi])])


def _combine_fft(arrays, mode: str):
    """Blend equal-shape stems in the spectrogram domain (UVR-style).

    Magnitudes are combined per time-frequency bin — ``avg_fft`` averages them,
    ``min_fft`` keeps the minimum (suppressing artifacts only one model produced).
    Reconstruction uses the phase of the COMPLEX SUM of the stems: in a region
    where one stem is only zero-padding, the sum's phase falls back to the stem
    that actually has energy there (using stem 0's phase alone would resynthesize
    the other stem's tail with constant zero phase — i.e. garble it).

    The window is clamped to the input length so sub-window stems don't crash,
    and everything runs in float32/complex64 to halve peak memory on long tracks.
    """
    import numpy as np  # type: ignore
    from scipy.signal import istft, stft  # type: ignore

    n, ch = arrays[0].shape
    nper = min(4096, n)
    nov = nper * 3 // 4
    out = np.zeros((n, ch), dtype=np.float32)
    for c in range(ch):
        specs = [stft(a[:, c].astype(np.float32), nperseg=nper, noverlap=nov)[2] for a in arrays]
        mags = np.stack([np.abs(z) for z in specs])
        mag = mags.mean(axis=0) if mode == "avg_fft" else mags.min(axis=0)
        phase = np.angle(np.sum(specs, axis=0))
        _, rec = istft((mag * np.exp(1j * phase)).astype(np.complex64), nperseg=nper, noverlap=nov)
        out[:, c] = rec[:n] if len(rec) >= n else np.pad(rec, (0, n - len(rec)))
    return out


def average_stems(
    paths: list[str | Path],
    out_path: str | Path,
    align: bool = True,
    algorithm: str = "avg_wave",
) -> Path:
    """Blend several stem files into one, time-aligned.

    Different separators emit the same transient a few samples apart, so we
    cross-correlate each stem against the first and shift it into phase before
    blending — otherwise the kick/snare attacks comb-filter and cancel (the
    opposite of the intended bleed reduction).

    ``algorithm``: "avg_wave" (waveform mean), or spectrogram-domain "avg_fft" /
    "min_fft" (UVR-style; min suppresses model-specific artifacts hardest).
    Shorter stems are zero-padded to the longest; the result is attenuated only
    if it would clip (never boosted, so levels stay comparable to the plain
    Demucs stem).
    """
    try:
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Stem averaging", "soundfile", extra="drums") from exc

    if not paths:
        raise ValueError("average_stems needs at least one path")
    if algorithm not in ("avg_wave", "avg_fft", "min_fft"):
        raise ValueError(f"Unknown ensemble algorithm: {algorithm!r}")

    arrays = []
    sr = None
    for p in paths:
        y, file_sr = sf.read(str(p), always_2d=True)  # (samples, channels)
        sr = sr or file_sr
        if file_sr != sr:
            raise ExternalToolError(f"Sample-rate mismatch averaging stems: {file_sr} vs {sr}")
        arrays.append(y)

    # Zero-pad to the longest (UVR-style) instead of discarding the tail, and
    # broadcast mono up to the widest channel count.
    max_len = max(a.shape[0] for a in arrays)
    max_ch = max(a.shape[1] for a in arrays)
    norm = []
    for a in arrays:
        if a.shape[0] < max_len:
            log.debug("Padding stem by %d samples to match the longest", max_len - a.shape[0])
            a = np.pad(a, ((0, max_len - a.shape[0]), (0, 0)))
        if a.shape[1] == 1 and max_ch > 1:
            a = np.repeat(a, max_ch, axis=1)  # broadcast true mono
        elif a.shape[1] != max_ch:
            # Anything else (e.g. stereo vs 4-channel) has no obviously-correct
            # mapping; failing beats silently discarding channels.
            raise ExternalToolError(f"Channel-count mismatch blending stems: {a.shape[1]} vs {max_ch}")
        norm.append(a)
    arrays = norm

    if align and len(arrays) > 1:
        ref = arrays[0].mean(axis=1)
        max_lag = int(0.05 * sr)  # expect only a few-ms offset; cap at 50 ms
        for i in range(1, len(arrays)):
            lag = _best_lag(ref, arrays[i].mean(axis=1), max_lag)
            if lag:
                arrays[i] = np.roll(arrays[i], lag, axis=0)
                log.info("Aligned stem %d by %+d samples (%.1f ms)", i, lag, 1000.0 * lag / sr)

    if algorithm == "avg_wave":
        acc = np.stack(arrays).mean(axis=0)
    else:
        acc = _combine_fft(arrays, algorithm)

    # Attenuate-only (like demucs save_audio's rescale): never boost a quiet stem.
    peak = float(np.max(np.abs(acc)))
    if peak > 0.98:
        acc = acc * (0.98 / peak)
    acc = acc.astype("float32")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), acc, sr)
    log.info("Blended %d stems (%s) -> %s", len(paths), algorithm, out_path)
    return out_path


def audio_separator_drums(mix_path: str | Path, out_dir: str | Path, model: str) -> Path:
    """Produce a drums stem from ``mix_path`` using python-audio-separator.

    ``model`` is an audio-separator model filename (e.g. a RoFormer or SCNet
    drum checkpoint). Requires ``pip install audio-separator`` and network access
    to fetch the model on first use. Returns the path to the drums stem.
    """
    exe = shutil.which("audio-separator")
    if exe is None:
        raise MissingDependencyError("Ensemble second model", "audio-separator", extra="ensemble")

    # Start from a clean temp dir so a stale drums stem from a previous run
    # can't be mistaken for this run's output.
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe, str(mix_path),
        "--model_filename", model,
        "--output_dir", str(out_dir),
        "--single_stem", "Drums",
        # Force WAV: recent python-audio-separator defaults the CLI to FLAC,
        # which our glob (and the averaging step) would otherwise miss.
        "--output_format", "WAV",
    ]
    log.info("Running audio-separator: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ExternalToolError(f"audio-separator failed ({proc.returncode}): {proc.stderr.strip()[:400]}")

    # Accept either extension defensively; pick the most recently written match.
    candidates = list(out_dir.glob("*[Dd]rums*.wav")) + list(out_dir.glob("*[Dd]rums*.flac"))
    if not candidates:
        raise ExternalToolError("audio-separator produced no drums stem.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def ensemble_drums(
    mix_path: str | Path,
    demucs_drums: str | Path,
    out_dir: str | Path,
    model: str,
    algorithm: str = "avg_wave",
) -> Path:
    """Blend the Demucs drums with a second model's drums, returning the blended stem.

    Falls back to the Demucs drums unchanged if the second model is unavailable,
    so callers can request the upgrade without hard-failing when it isn't set up.
    """
    out_dir = Path(out_dir)
    try:
        second = audio_separator_drums(mix_path, out_dir / "ensemble_tmp", model)
    except (MissingDependencyError, ExternalToolError) as exc:
        log.warning("Ensemble skipped (%s); using Demucs drums as-is.", exc)
        return Path(demucs_drums)
    return average_stems([demucs_drums, second], out_dir / "drums_ensemble.wav", algorithm=algorithm)
