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
from .errors import AudioLoadError, ExternalToolError, MissingDependencyError
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


def _demucs_progress_fraction(d: dict) -> float | None:
    """Overall 0..1 fraction from a demucs.api callback dict (best effort)."""
    try:
        models = max(1, int(d.get("models", 1) or 1))
        length = float(d.get("audio_length", 0) or 0)
        offset = float(d.get("segment_offset", 0) or 0)
        within = min(1.0, offset / length) if length > 0 else 0.0
        frac = (int(d.get("model_idx_in_bag", 0) or 0) + within) / models
        return min(1.0, max(0.0, frac))
    except Exception:
        return None


def separate(
    audio_path: str | Path,
    out_dir: str | Path,
    config: SeparationConfig | None = None,
    progress=None,
) -> Stems:
    """Separate ``audio_path`` into stems under ``out_dir``.

    Returns a :class:`Stems` object with paths to the stems requested in
    ``config.stems`` (drums + bass by default). ``progress``, if given, is
    called with a 0..1 fraction as inference advances (API path only).
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
        return _separate_api(audio_path, out_dir, config, device, progress)
    except MissingDependencyError:
        raise
    except (ImportError, AttributeError) as exc:
        # Only an API-shape mismatch (a Demucs version whose api module differs)
        # justifies retrying via the CLI. Genuine runtime failures — a bad model
        # name, undecodable input, OOM, or a blocked weights download — would
        # fail identically through the CLI, so let them surface directly instead
        # of masking the root cause behind a second doomed run.
        log.warning("Demucs Python API unavailable (%s); trying the CLI.", exc)
        return _separate_cli(audio_path, out_dir, config, device)


def _separate_api(audio_path: Path, out_dir: Path, config: SeparationConfig, device: str, progress=None) -> Stems:
    try:
        from demucs.api import Separator  # type: ignore
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Source separation", "demucs", extra="separation") from exc

    callback = None
    if progress is not None:
        def callback(d):  # noqa: ANN001 - demucs callback dict
            frac = _demucs_progress_fraction(d if isinstance(d, dict) else {})
            if frac is not None:
                try:
                    progress(frac)
                except Exception:  # a UI callback must never break separation
                    pass

    separator = Separator(
        model=config.model,
        device=device,
        shifts=config.shifts,
        overlap=config.overlap,
        segment=config.segment,  # None -> model default; lower to save GPU memory
        jobs=config.jobs,
        callback=callback,
    )
    _origin, separated = separator.separate_audio_file(str(audio_path))
    return _save_stems(separated, out_dir, config, separator.samplerate)


def _save_stems(separated: dict, out_dir: Path, config: SeparationConfig, samplerate: int) -> Stems:
    """Write the requested stems from a Demucs ``{name: tensor}`` dict to disk.

    Split out from :func:`_separate_api` so the tensor->file plumbing can be
    tested against the real ``demucs.api.save_audio`` without downloading model
    weights.
    """
    from demucs.api import save_audio  # type: ignore

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = Stems()
    ext = "mp3" if config.mp3 else "wav"
    for name, source in separated.items():
        if name not in config.stems:
            continue
        stem_path = out_dir / f"{name}.{ext}"
        save_kwargs = {"bitrate": config.mp3_bitrate} if config.mp3 else {}
        save_audio(source, str(stem_path), samplerate=samplerate, **save_kwargs)
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
    if config.segment:
        cmd += ["--segment", str(config.segment)]
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
    except subprocess.CalledProcessError as exc:
        raise ExternalToolError(
            f"Demucs CLI failed (exit {exc.returncode}). Check the model name "
            f"('{config.model}') and that the input is a decodable audio file."
        ) from exc

    # Demucs CLI writes to <out_dir>/<model>/<track_name>/<stem>.<ext>. Move the
    # requested stems up to the flat <out_dir>/<stem>.<ext> layout so they match
    # the API path (_save_stems) and are found by _discover_existing_stems when a
    # later run reuses stems with do_separation=False.
    import shutil

    ext = "mp3" if config.mp3 else "wav"
    track_dir = out_dir / config.model / audio_path.stem
    stems = Stems()
    for name in config.stems:
        candidate = track_dir / f"{name}.{ext}"
        if candidate.exists():
            flat = out_dir / f"{name}.{ext}"
            if candidate.resolve() != flat.resolve():
                shutil.move(str(candidate), str(flat))
            setattr(stems, name, flat)
            log.info("  -> %s", flat)
    return stems
