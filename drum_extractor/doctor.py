"""`drum-extractor doctor` — environment checkup.

Probes every dependency and external tool the pipeline can use, and prints a
table of what's ready, what's missing, and the exact command that fixes each
gap. Designed so a fresh user can go from "why doesn't X work" to the right
install command without reading source.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass

from .logging_utils import get_logger

log = get_logger(__name__)

OK, MISSING, INFO = "ok", "missing", "info"


@dataclass
class Check:
    feature: str
    status: str  # ok | missing | info
    detail: str
    fix: str = ""
    core: bool = False


def _probe_import(module: str) -> tuple[bool, str]:
    """Really import (find_spec would miss broken installs), but quietly:
    TF banners and deprecation warnings would otherwise bury the report."""
    import contextlib
    import io
    import warnings

    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            __import__(module)
        return True, _version_of(module)
    except Exception as exc:  # ImportError or any init failure
        return False, type(exc).__name__


def _version_of(module: str) -> str:
    """Installed version via package metadata (many modules lack __version__)."""
    from importlib import metadata

    dist_names = {"basic_pitch": "basic-pitch", "guitarpro": "PyGuitarPro"}
    for name in (dist_names.get(module, module), module.replace("_", "-")):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    import sys as _sys

    return str(getattr(_sys.modules.get(module), "__version__", ""))


def _probe_cmd(*names: str) -> str | None:
    for name in names:
        if shutil.which(name):
            return name
    return None


def run_doctor() -> list[Check]:
    checks: list[Check] = []

    # --- Python itself ---
    py_ok = sys.version_info >= (3, 10)
    checks.append(
        Check("Python >= 3.10", OK if py_ok else MISSING,
              f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
              "install Python 3.10+ from python.org", core=True)
    )

    # --- Core ---
    for mod, pkg in (("numpy", "numpy"), ("pretty_midi", "pretty_midi")):
        ok, v = _probe_import(mod)
        checks.append(Check(f"core: {pkg}", OK if ok else MISSING, v, f"pip install {pkg}", core=True))

    # --- Stage 1: separation ---
    ok, v = _probe_import("demucs")
    checks.append(Check("separation: demucs", OK if ok else MISSING, v, 'pip install "drum-extractor[separation]"'))
    if ok:
        try:
            import torch

            if torch.cuda.is_available():
                dev = f"CUDA ({torch.cuda.get_device_name(0)})"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                dev = "Apple MPS"
            else:
                dev = "CPU only (works; roughly the track's length per song)"
            checks.append(Check("separation: compute device", INFO, dev))
        except Exception:
            pass

    # --- Stage 2a: drum transcription ---
    for mod, extra in (("librosa", "drums"), ("soundfile", "drums"), ("scipy", "drums")):
        ok, v = _probe_import(mod)
        checks.append(Check(f"drums: {mod}", OK if ok else MISSING, v, f'pip install "drum-extractor[{extra}]"'))
    adtof = _probe_cmd("adtof")
    checks.append(
        Check("drums: ADTOF (best transcriber)", OK if adtof else MISSING,
              adtof or "onset fallback will be used",
              'pip install "drum-extractor[adtof]"  (installs ADTOF-pytorch from git)')
    )

    # --- Stage 2b: bass ---
    ok, v = _probe_import("basic_pitch")
    checks.append(Check("bass: basic-pitch", OK if ok else MISSING, v, 'pip install "drum-extractor[bass]"'))
    ok, v = _probe_import("torchcrepe")
    checks.append(Check("bass: torchcrepe (octave refine)", OK if ok else MISSING, v, 'pip install "drum-extractor[bass-crepe]"'))

    # --- Stage 3: quantization ---
    ok, v = _probe_import("madmom")
    checks.append(
        Check("quantize: madmom", OK if ok else INFO,
              v if ok else "librosa fallback will be used (fine for most songs)",
              'pip install "drum-extractor[quantize]"  (needs a C toolchain; optional)')
    )

    # --- Stage 4: notation ---
    ok, v = _probe_import("music21")
    checks.append(Check("notation: music21", OK if ok else MISSING, v, 'pip install "drum-extractor[notation]"'))
    # Resolve through the SAME lookup render_pdf uses, so doctor can never say
    # "ok" for a MuseScore binary the renderer wouldn't find.
    from .notation import find_musescore

    mscore = find_musescore()
    checks.append(
        Check("notation: MuseScore (PDF export)", OK if mscore else INFO,
              mscore or "MusicXML still produced; PDF needs MuseScore",
              "install MuseScore 4 from musescore.org and put its CLI on PATH")
    )

    # --- Guitar Pro export ---
    ok, v = _probe_import("guitarpro")
    checks.append(
        Check("tabs: PyGuitarPro (.gp5 export)", OK if ok else INFO,
              v if ok else "ASCII tabs still produced; .gp5 needs PyGuitarPro",
              'pip install "drum-extractor[gp]"')
    )

    # --- Web UI ---
    ok, v = _probe_import("flask")
    checks.append(Check("web: flask", OK if ok else MISSING, v, 'pip install "drum-extractor[web]"'))
    ok, v = _probe_import("verovio")
    checks.append(
        Check("web: verovio (inline sheet SVG)", OK if ok else INFO,
              v if ok else "sheet preview falls back to PDF/downloads",
              'pip install "drum-extractor[web]"')
    )

    # --- Extras ---
    aud = _probe_cmd("audio-separator")
    checks.append(
        Check("ensemble: audio-separator", OK if aud else INFO,
              aud or "only needed for --ensemble-model",
              'pip install "drum-extractor[ensemble]"')
    )
    ffmpeg = _probe_cmd("ffmpeg")
    checks.append(
        Check("ffmpeg (broad format support)", OK if ffmpeg else INFO,
              ffmpeg or "wav/flac/mp3 work without it; helps with m4a/aac",
              "install ffmpeg via your package manager")
    )

    return checks


def format_report(checks: list[Check]) -> str:
    mark = {OK: "[ok]", MISSING: "[--]", INFO: "[..]"}
    width = max(len(c.feature) for c in checks) + 2
    lines = ["drum-extractor doctor", "=" * 60]
    for c in checks:
        lines.append(f"{mark[c.status]} {c.feature:<{width}} {c.detail}")
        if c.status == MISSING and c.fix:
            lines.append(f"     {'':<{width}} fix: {c.fix}")
    missing_core = [c for c in checks if c.core and c.status == MISSING]
    missing_opt = [c for c in checks if not c.core and c.status == MISSING]
    lines.append("=" * 60)
    if missing_core:
        lines.append(f"BROKEN: {len(missing_core)} core requirement(s) missing — fix those first.")
    elif missing_opt:
        lines.append(
            f"Usable. {len(missing_opt)} optional feature(s) unavailable — the pipeline degrades "
            "gracefully around them; install their extras when you want them."
        )
    else:
        lines.append("Everything is ready.")
    return "\n".join(lines)


def doctor_main() -> int:
    checks = run_doctor()
    print(format_report(checks))
    return 1 if any(c.core and c.status == MISSING for c in checks) else 0
