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


def average_stems(paths: list[str | Path], out_path: str | Path) -> Path:
    """Average several mono/stereo stem files into one, aligned to the shortest.

    Files are matched on channel count (mono is broadcast); the result is
    peak-normalised. Used to blend two separators' drum stems.
    """
    try:
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Stem averaging", "soundfile", extra="drums") from exc

    if not paths:
        raise ValueError("average_stems needs at least one path")

    arrays = []
    sr = None
    for p in paths:
        y, file_sr = sf.read(str(p), always_2d=True)  # (samples, channels)
        sr = sr or file_sr
        if file_sr != sr:
            raise ExternalToolError(f"Sample-rate mismatch averaging stems: {file_sr} vs {sr}")
        arrays.append(y)

    min_len = min(a.shape[0] for a in arrays)
    max_ch = max(a.shape[1] for a in arrays)
    acc = np.zeros((min_len, max_ch), dtype=np.float64)
    for a in arrays:
        a = a[:min_len]
        if a.shape[1] == 1 and max_ch > 1:
            a = np.repeat(a, max_ch, axis=1)
        acc[:, : a.shape[1]] += a
    acc /= len(arrays)

    peak = float(np.max(np.abs(acc))) or 1.0
    acc = (acc / peak * 0.98).astype("float32")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), acc, sr)
    log.info("Averaged %d stems -> %s", len(paths), out_path)
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


def ensemble_drums(mix_path: str | Path, demucs_drums: str | Path, out_dir: str | Path, model: str) -> Path:
    """Blend the Demucs drums with a second model's drums, returning the averaged stem.

    Falls back to the Demucs drums unchanged if the second model is unavailable,
    so callers can request the upgrade without hard-failing when it isn't set up.
    """
    out_dir = Path(out_dir)
    try:
        second = audio_separator_drums(mix_path, out_dir / "ensemble_tmp", model)
    except (MissingDependencyError, ExternalToolError) as exc:
        log.warning("Ensemble skipped (%s); using Demucs drums as-is.", exc)
        return Path(demucs_drums)
    return average_stems([demucs_drums, second], out_dir / "drums_ensemble.wav")
