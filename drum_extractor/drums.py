"""Stage 2a — automatic drum transcription (drum stem -> drum hits).

Primary backend is ADTOF (best open model for real, distorted music, trained on
rock/metal rhythm-game charts). A dependency-free librosa onset+band-energy
fallback is provided so the pipeline produces a rough transcription before ADTOF
is installed — useful for smoke-testing the notation stage.

Accuracy reality: kick/snare/hi-hat on a mid-tempo groove is genuinely useful;
fast double-bass and blast beats are undercounted by every current model. Treat
fast-metal output as a scaffold to correct by ear.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .config import DrumTranscriptionConfig
from .errors import ExternalToolError
from .events import DrumHit
from .gm_drum_map import KICK, SNARE, HIHAT_CLOSED, normalize_instrument
from .logging_utils import get_logger

log = get_logger(__name__)


def transcribe_drums(drum_stem: str | Path, config: DrumTranscriptionConfig | None = None) -> list[DrumHit]:
    """Transcribe an isolated drum stem into a list of :class:`DrumHit`."""
    config = config or DrumTranscriptionConfig()
    drum_stem = Path(drum_stem)
    if not drum_stem.exists():
        raise ExternalToolError(f"Drum stem not found: {drum_stem}")

    if config.backend == "none":
        return []
    if config.backend == "adtof":
        try:
            hits = _transcribe_adtof(drum_stem, config)
        except ExternalToolError as exc:
            log.warning("ADTOF backend failed (%s); falling back to librosa onset detection.", exc)
            hits = _transcribe_onsets(drum_stem, config)
    elif config.backend == "onset":
        hits = _transcribe_onsets(drum_stem, config)
    else:
        raise ValueError(f"Unknown drum transcription backend: {config.backend!r}")

    if config.boost_double_kick:
        from .kick import boost_double_kick

        hits = boost_double_kick(hits, drum_stem, velocity=config.default_velocity)
    return hits


# --- ADTOF ---------------------------------------------------------------------------

def _transcribe_adtof(drum_stem: Path, config: DrumTranscriptionConfig) -> list[DrumHit]:
    """Run ADTOF via its CLI and parse the result.

    ADTOF installs vary; the exact command is configurable via
    ``config.adtof_command``. We accept either a MIDI or a text/CSV onset file as
    the output artifact and normalise the class labels to canonical names.
    """
    out_path = drum_stem.with_suffix(".adtof.txt")
    cmd = [part.format(input=str(drum_stem), output=str(out_path)) for part in config.adtof_command]
    log.info("Running ADTOF: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ExternalToolError(
            "ADTOF is not installed or not on PATH. Install with: pip install "
            '"drum-extractor[drums]"  (see README for the pip install adtof notes).'
        ) from exc
    if proc.returncode != 0:
        raise ExternalToolError(f"ADTOF exited {proc.returncode}: {proc.stderr.strip()[:500]}")

    # ADTOF may write the requested text file or a sibling MIDI; handle both.
    if out_path.exists():
        return _parse_onset_text(out_path, config.default_velocity)
    midi_candidate = drum_stem.with_suffix(".mid")
    if midi_candidate.exists():
        from .midi_io import read_drum_hits

        return read_drum_hits(midi_candidate, config.default_velocity)
    raise ExternalToolError("ADTOF ran but produced no recognizable output file.")


def _parse_onset_text(path: Path, velocity: int) -> list[DrumHit]:
    """Parse ``time<sep>label`` lines (tab/comma/space separated)."""
    hits: list[DrumHit] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", "\t").split()
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
        except ValueError:
            continue  # header row
        hits.append(DrumHit(time=t, instrument=normalize_instrument(parts[1]), velocity=velocity))
    hits.sort(key=lambda h: h.time)
    log.info("ADTOF: parsed %d drum hits", len(hits))
    return hits


# --- librosa onset fallback ----------------------------------------------------------

def _transcribe_onsets(drum_stem: Path, config: DrumTranscriptionConfig) -> list[DrumHit]:
    """Rough 3-way (kick/snare/hi-hat) transcription using onset detection + band energy.

    For each detected onset we measure energy in a low / mid / high band and emit
    a hit for every band that stands out. This can catch simultaneous kick+hi-hat,
    but it cannot distinguish toms/cymbals or fast double-kick — it exists so the
    downstream stages have data before ADTOF is installed.
    """
    try:
        import librosa  # type: ignore
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        from .errors import MissingDependencyError

        raise MissingDependencyError("Onset-based drum fallback", "librosa", extra="drums") from exc

    log.info("Transcribing drums with librosa onset fallback (rough kick/snare/hi-hat only).")
    y, sr = librosa.load(str(drum_stem), sr=44100, mono=True)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    n_fft = 2048
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low = freqs < 150
    mid = (freqs >= 150) & (freqs < 2000)
    high = freqs >= 6000

    hits: list[DrumHit] = []
    band_vals = {"low": [], "mid": [], "high": []}
    per_onset = []
    for t in onset_times:
        frame = librosa.time_to_frames(t, sr=sr, hop_length=512)
        frame = int(min(max(frame, 0), stft.shape[1] - 1))
        window = stft[:, frame : frame + 3].mean(axis=1) if frame + 3 <= stft.shape[1] else stft[:, frame]
        e_low = float(window[low].sum())
        e_mid = float(window[mid].sum())
        e_high = float(window[high].sum())
        per_onset.append((t, e_low, e_mid, e_high))
        band_vals["low"].append(e_low)
        band_vals["mid"].append(e_mid)
        band_vals["high"].append(e_high)

    # Adaptive thresholds: a band "fires" if it clears its own median.
    thr = {b: (np.median(v) if v else 0.0) for b, v in band_vals.items()}
    for t, e_low, e_mid, e_high in per_onset:
        fired = False
        if e_low >= thr["low"]:
            hits.append(DrumHit(time=float(t), instrument=KICK, velocity=config.default_velocity))
            fired = True
        if e_high >= thr["high"]:
            hits.append(DrumHit(time=float(t), instrument=HIHAT_CLOSED, velocity=config.default_velocity))
            fired = True
        # Snare when mid dominates, or as the default if nothing else fired.
        if e_mid >= thr["mid"] and e_mid >= e_low:
            hits.append(DrumHit(time=float(t), instrument=SNARE, velocity=config.default_velocity))
            fired = True
        if not fired:
            hits.append(DrumHit(time=float(t), instrument=SNARE, velocity=config.default_velocity))

    hits.sort(key=lambda h: h.time)
    log.info("Onset fallback: %d onsets -> %d hits", len(onset_times), len(hits))
    return hits
