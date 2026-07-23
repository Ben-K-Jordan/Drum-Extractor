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
    out_path = drum_stem.with_suffix(".adtof.mid")
    cmd = [part.format(input=str(drum_stem), output=str(out_path)) for part in config.adtof_command]
    log.info("Running ADTOF: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ExternalToolError(
            "ADTOF was not found on PATH. Note there is no `pip install adtof` that provides a CLI: "
            "install ADTOF-pytorch (which provides the `adtof` command) with "
            "`pip install \"drum-extractor[adtof]\"`, or point DrumTranscriptionConfig.adtof_command at "
            "your own install / use the librosa 'onset' backend. See the README."
        ) from exc
    if proc.returncode != 0:
        raise ExternalToolError(f"ADTOF exited {proc.returncode}: {proc.stderr.strip()[:500]}")

    return _read_adtof_output(out_path, drum_stem, config.default_velocity)


def _read_adtof_output(out_path: Path, drum_stem: Path, velocity: int) -> list[DrumHit]:
    """Read whatever ADTOF wrote, detecting MIDI vs text by content.

    ADTOF variants differ: ADTOF-pytorch writes MIDI (bytes are MIDI regardless
    of the output filename), while the original writes a ``time\\tpitch`` text
    file — sometimes into a folder named by track title rather than the exact
    path we asked for. So we search the requested path plus sibling ``.mid``/
    ``.txt`` files and dispatch by the file's actual content, not its extension.
    """
    from .midi_io import read_drum_hits

    candidates = [out_path]
    candidates += sorted(out_path.parent.glob(f"{drum_stem.stem}*.mid"))
    candidates += sorted(out_path.parent.glob(f"{drum_stem.stem}*.txt"))
    seen = set()
    for p in candidates:
        rp = p.resolve()
        if rp in seen or not p.exists() or p.stat().st_size == 0:
            continue
        seen.add(rp)
        if p.read_bytes()[:4] == b"MThd":  # standard MIDI header
            hits = read_drum_hits(p, velocity)
        else:
            try:
                hits = _parse_onset_text(p, velocity)
            except (UnicodeDecodeError, ValueError):
                continue
        if hits:
            return hits
    raise ExternalToolError("ADTOF ran but produced no recognizable output (.mid or time/pitch .txt).")


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
    # Contiguous bands: the old mid (150-2000) / high (>=6000) split left a
    # 2-6 kHz hole where snare crack and stick attack live, so that energy was
    # invisible to the classifier.
    low = freqs < 150
    mid = (freqs >= 150) & (freqs < 5000)
    high = freqs >= 5000

    hits: list[DrumHit] = []
    vel = config.default_velocity
    # Dominance-based classification: the strongest band ALWAYS fires (every
    # onset gets exactly one primary instrument), and a non-dominant band fires
    # only if its energy share clears its own (deliberately high) secondary
    # threshold — i.e. only when the onset genuinely looks like two instruments
    # struck together (kick+hat, snare+crash). Firing every band that clears a
    # low absolute share stamps a phantom second instrument on nearly every
    # onset, because real drums spread energy across bands (a snare has body
    # AND wires; a kick has a beater click).
    band_instr = {"low": KICK, "mid": SNARE, "high": HIHAT_CLOSED}
    secondary = {
        "low": config.onset_kick_share,
        "mid": config.onset_snare_share,
        "high": config.onset_hihat_share,
    }
    for t in onset_times:
        frame = librosa.time_to_frames(t, sr=sr, hop_length=512)
        frame = int(min(max(frame, 0), stft.shape[1] - 1))
        window = stft[:, frame : frame + 3].mean(axis=1) if frame + 3 <= stft.shape[1] else stft[:, frame]
        e_low = float(window[low].sum())
        e_mid = float(window[mid].sum())
        e_high = float(window[high].sum())
        total = e_low + e_mid + e_high + 1e-12
        shares = {"low": e_low / total, "mid": e_mid / total, "high": e_high / total}

        dominant = max(shares, key=shares.get)  # type: ignore[arg-type]
        hits.append(DrumHit(time=float(t), instrument=band_instr[dominant], velocity=vel))
        for band, share in shares.items():
            if band != dominant and share >= secondary[band]:
                hits.append(DrumHit(time=float(t), instrument=band_instr[band], velocity=vel))

    hits.sort(key=lambda h: h.time)
    log.info("Onset fallback: %d onsets -> %d hits", len(onset_times), len(hits))
    return hits
