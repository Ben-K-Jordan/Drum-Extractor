"""Stage 2b — bass transcription (bass stem -> notes + tab).

Bass is the easiest transcription task here because the line is mostly
monophonic. Spotify's basic-pitch turns the isolated bass stem into note-level
MIDI; an optional torchcrepe pass octave-corrects the notes distorted metal bass
tends to send an octave high. A simple cost-minimising mapper then assigns each
note a string+fret for tab.
"""

from __future__ import annotations

from pathlib import Path

from .config import BassTranscriptionConfig
from .errors import MissingDependencyError
from .events import BassNote
from .logging_utils import get_logger

log = get_logger(__name__)


def transcribe_bass(bass_stem: str | Path, config: BassTranscriptionConfig | None = None) -> list[BassNote]:
    """Transcribe an isolated bass stem into notes, with string/fret assigned."""
    config = config or BassTranscriptionConfig()
    bass_stem = Path(bass_stem)
    if config.backend == "none":
        return []
    if config.backend != "basic_pitch":
        raise ValueError(f"Unknown bass transcription backend: {config.backend!r}")

    try:
        from basic_pitch.inference import predict  # type: ignore
        from basic_pitch import ICASSP_2022_MODEL_PATH  # type: ignore
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Bass transcription", "basic-pitch", extra="bass") from exc

    log.info("Transcribing bass with basic-pitch (%.0f-%.0f Hz)", config.min_frequency, config.max_frequency)
    _model_output, midi_data, _note_events = predict(
        str(bass_stem),
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        minimum_frequency=config.min_frequency,
        maximum_frequency=config.max_frequency,
    )

    notes: list[BassNote] = []
    for inst in midi_data.instruments:
        for n in inst.notes:
            notes.append(BassNote(start=float(n.start), end=float(n.end), pitch=int(n.pitch), velocity=int(n.velocity)))
    notes.sort(key=lambda n: (n.start, n.pitch))

    if config.refine_with_crepe:
        notes = _refine_octaves_with_crepe(bass_stem, notes, config.min_frequency)

    assign_tab(notes, config)
    log.info("Bass: %d notes transcribed", len(notes))
    return notes


def _refine_octaves_with_crepe(bass_stem: Path, notes: list[BassNote], min_frequency: float = 32.7) -> list[BassNote]:
    """Correct octave errors by comparing each note to torchcrepe's F0 estimate.

    Distorted bass makes pitch trackers latch onto a harmonic (usually an octave
    up). We compare basic-pitch's pitch to the median CREPE fundamental over the
    note; if CREPE is confidently ~12 semitones lower, we pull the note down.

    ``fmin`` must not go below torchcrepe's lowest model bin (~31.8 Hz): a lower
    fmin pins the Viterbi decode to bin 0 and collapses the whole F0 track to a
    constant, so we clamp to at least C1 (32.7 Hz).
    """
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        import torchcrepe  # type: ignore
        import librosa  # type: ignore
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("CREPE octave refinement", "torchcrepe", extra="bass-crepe") from exc

    fmin = max(32.7, float(min_frequency))
    y, sr = librosa.load(str(bass_stem), sr=16000, mono=True)
    audio = torch.tensor(y)[None]
    hop = 160  # 10 ms at 16 kHz
    f0 = torchcrepe.predict(audio, sr, hop_length=hop, fmin=fmin, fmax=500, model="full", batch_size=512, device="cpu")
    f0 = f0[0].cpu().numpy()
    times = np.arange(len(f0)) * hop / sr

    def crepe_pitch(start: float, end: float) -> float | None:
        mask = (times >= start) & (times < end)
        seg = f0[mask]
        seg = seg[seg > 0]
        if seg.size == 0:
            return None
        return float(librosa.hz_to_midi(np.median(seg)))

    corrected = 0
    for n in notes:
        cp = crepe_pitch(n.start, n.end)
        if cp is None:
            continue
        diff = n.pitch - cp
        if 10.5 <= diff <= 13.5:  # basic-pitch ~1 octave above CREPE
            n.pitch -= 12
            corrected += 1
    if corrected:
        log.info("CREPE refinement: corrected %d octave errors", corrected)
    return notes


def assign_tab(notes: list[BassNote], config: BassTranscriptionConfig) -> None:
    """Assign a (string, fret) to each note, minimising hand movement.

    Greedy: for each note pick the playable string/fret closest to the previous
    note's fret. For 4 strings and mostly single notes this yields clean,
    playable tab — the tab mapping is the easy part; note accuracy is the limit.
    """
    tuning = config.tuning
    prev_fret = 5  # start hand around the 5th fret
    unreachable = 0
    for n in notes:
        best: tuple[int, int] | None = None
        best_cost = 1e9
        for s, open_pitch in enumerate(tuning):
            fret = n.pitch - open_pitch
            if 0 <= fret <= config.frets:
                cost = abs(fret - prev_fret) + (0.5 if fret == 0 else 0.0)  # slight bias against open strings mid-phrase
                if cost < best_cost:
                    best_cost = cost
                    best = (s, fret)
        if best is not None:
            n.string, n.fret = best
            prev_fret = best[1]
        else:
            unreachable += 1
    if unreachable:
        log.warning(
            "%d bass note(s) fall outside the fretboard range (%s); they are marked 'x' in the tab "
            "but retain their true pitch in the MIDI.",
            unreachable,
            f"tuning low={tuning[0]}, {config.frets} frets",
        )


def render_ascii_tab(notes: list[BassNote], config: BassTranscriptionConfig, columns: int = 80) -> str:
    """Render a simple ASCII bass tab (one horizontal block).

    This is a readable text preview, not engraved notation — good enough to
    practise from and to sanity-check the transcription.
    """
    tuning = config.tuning
    string_names = _string_names(tuning)
    lanes: list[list[str]] = [[] for _ in tuning]
    for n in notes:
        if n.string is None or n.fret is None:
            # Out-of-range note: keep it visible (and count-consistent with the
            # MIDI) by marking it 'x' on the lowest string rather than dropping it.
            token, target = "x", 0
        else:
            token, target = str(n.fret), n.string
        width = max(len(token) + 1, 3)
        for s in range(len(tuning)):
            if s == target:
                lanes[s].append(token.rjust(width - 1, "-") + "-")
            else:
                lanes[s].append("-" * width)

    # Right-justify string labels to a common width so rows with a 2-char
    # accidental name (e.g. 'F#') stay column-aligned with 1-char names.
    label_w = max((len(nm) for nm in string_names), default=1)
    lines = []
    for s in range(len(tuning) - 1, -1, -1):  # highest string on top
        body = "".join(lanes[s]) or "-" * 8
        lines.append(f"{string_names[s].rjust(label_w)}|-{body}")
    return "\n".join(lines)


def _string_names(tuning: tuple[int, ...]) -> list[str]:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return [names[p % 12] for p in tuning]
