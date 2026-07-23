"""Stage 1 — source separation with Demucs.

Given a full-mix song, produce isolated stems (drums + bass by default). This is
the mature, "solved" half of the project: Demucs v4 (htdemucs_ft) is MIT-licensed
and gives clean drums/bass on dense metal mixes.

Uses the Demucs Python API (``demucs.api.Separator``) when available and falls
back to the ``python -m demucs`` CLI, so it works across Demucs installs.
"""

from __future__ import annotations

from pathlib import Path

from .config import SeparationConfig
from .errors import AudioLoadError, MissingDependencyError
from .events import Stems
from .logging_utils import get_logger

log = get_logger(__name__)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ModuleNotFoundError:
        pass
    return "cpu"


def separate(audio_path: str | Path, out_dir: str | Path, config: SeparationConfig | None = None) -> Stems:
    """Separate ``audio_path`` into stems under ``out_dir``.

    Returns a :class:`Stems` object with paths to the stems requested in
    ``config.stems`` (drums + bass by default).
    """
    config = config or SeparationConfig()
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    if not audio_path.exists():
        raise AudioLoadError(f"Input audio not found: {audio_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(config.device)
    log.info("Separating %s with %s on %s", audio_path.name, config.model, device)
    if device == "cpu":
        log.warning("Running Demucs on CPU — expect roughly 1-2x the track length per song.")

    try:
        return _separate_api(audio_path, out_dir, config, device)
    except MissingDependencyError:
        raise
    except Exception as exc:  # pragma: no cover - fall back to CLI on API quirks
        log.warning("Demucs Python API path failed (%s); falling back to CLI.", exc)
        return _separate_cli(audio_path, out_dir, config, device)


def _separate_api(audio_path: Path, out_dir: Path, config: SeparationConfig, device: str) -> Stems:
    try:
        from demucs.api import Separator, save_audio  # type: ignore
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Source separation", "demucs", extra="separation") from exc

    separator = Separator(
        model=config.model,
        device=device,
        shifts=config.shifts,
        overlap=config.overlap,
        jobs=config.jobs,
    )
    _origin, separated = separator.separate_audio_file(str(audio_path))

    stems = Stems()
    ext = "mp3" if config.mp3 else "wav"
    for name, source in separated.items():
        if name not in config.stems:
            continue
        stem_path = out_dir / f"{name}.{ext}"
        save_kwargs = {}
        if config.mp3:
            save_kwargs.update(bitrate=config.mp3_bitrate)
        save_audio(source, str(stem_path), samplerate=separator.samplerate, **save_kwargs)
        setattr(stems, name, stem_path)
        log.info("  -> %s", stem_path)
    return stems


def _separate_cli(audio_path: Path, out_dir: Path, config: SeparationConfig, device: str) -> Stems:
    import subprocess
    import sys

    two_stems = None
    if set(config.stems) == {"drums"}:
        two_stems = "drums"
    elif set(config.stems) == {"bass"}:
        two_stems = "bass"

    cmd = [
        sys.executable, "-m", "demucs",
        "-n", config.model,
        "-d", device,
        "--shifts", str(config.shifts),
        "--overlap", str(config.overlap),
        "-o", str(out_dir),
    ]
    if config.mp3:
        cmd += ["--mp3", "--mp3-bitrate", str(config.mp3_bitrate)]
    if two_stems:
        cmd += ["--two-stems", two_stems]
    cmd.append(str(audio_path))

    log.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise MissingDependencyError("Source separation", "demucs", extra="separation") from exc

    # Demucs CLI writes to <out_dir>/<model>/<track_name>/<stem>.<ext>
    ext = "mp3" if config.mp3 else "wav"
    track_dir = out_dir / config.model / audio_path.stem
    stems = Stems()
    for name in config.stems:
        candidate = track_dir / f"{name}.{ext}"
        if candidate.exists():
            setattr(stems, name, candidate)
            log.info("  -> %s", candidate)
    return stems
