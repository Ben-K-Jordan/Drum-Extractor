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

    notes = basic_pitch_notes(
        bass_stem,
        feature="Bass transcription",
        min_frequency=config.min_frequency,
        max_frequency=config.max_frequency,
        minimum_note_length_ms=config.minimum_note_length_ms,
        onset_threshold=config.onset_threshold,
        frame_threshold=config.frame_threshold,
    )

    if config.refine_with_crepe:
        notes = _refine_octaves_with_crepe(bass_stem, notes, config.min_frequency)

    assign_tab(notes, config)
    log.info("Bass: %d notes transcribed", len(notes))
    return notes


def basic_pitch_notes(
    stem: str | Path,
    feature: str,
    min_frequency: float,
    max_frequency: float,
    minimum_note_length_ms: float,
    onset_threshold: float,
    frame_threshold: float,
) -> list[BassNote]:
    """Run basic-pitch on ``stem`` and return sorted note events.

    Shared by the bass and guitar paths — basic-pitch is polyphonic, so the
    same engine reads chords for guitar.
    """
    try:
        from basic_pitch.inference import predict  # type: ignore
        from basic_pitch import ICASSP_2022_MODEL_PATH  # type: ignore
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(feature, "basic-pitch", extra="bass") from exc

    log.info("%s with basic-pitch (%.0f-%.0f Hz)", feature, min_frequency, max_frequency)
    _model_output, midi_data, _note_events = predict(
        str(stem),
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
        minimum_note_length=minimum_note_length_ms,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
    )

    notes: list[BassNote] = []
    for inst in midi_data.instruments:
        for n in inst.notes:
            notes.append(BassNote(start=float(n.start), end=float(n.end), pitch=int(n.pitch), velocity=int(n.velocity)))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def _refine_octaves_with_crepe(bass_stem: Path, notes: list[BassNote], min_frequency: float = 32.7) -> list[BassNote]:
    """Correct octave errors by comparing each note to torchcrepe's F0 estimate.

    Distorted bass makes pitch trackers latch onto a harmonic (usually an octave
    up). We compare basic-pitch's pitch to the median CREPE fundamental over the
    note; if CREPE is confidently ~12 semitones lower, we pull the note down.

    ``fmin`` must not go below torchcrepe's lowest model bin (~31.8 Hz):
    empirically an out-of-range fmin degrades the decode badly (the track can
    collapse toward the lowest bin), so we clamp to at least C1 (32.7 Hz). Note
    this also bounds 5-string low-B (30.9 Hz) detection — CREPE simply cannot
    see below its bottom bin.
    """
    if not notes:
        return notes  # nothing to refine; skip the (expensive) CREPE decode

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
    pitch, periodicity = torchcrepe.predict(
        audio, sr, hop_length=hop, fmin=fmin, fmax=500, model="full",
        batch_size=512, device="cpu", return_periodicity=True,
    )
    # predict() emits a pitch for EVERY frame (incl. silence/decay/unvoiced), so
    # a raw median is polluted. Threshold on smoothed periodicity — torchcrepe's
    # documented recipe — so only confidently-voiced frames feed the median.
    periodicity = torchcrepe.filter.median(periodicity, 3)
    pitch = torchcrepe.threshold.At(0.21)(pitch, periodicity)
    f0 = pitch[0].cpu().numpy()
    times = np.arange(len(f0)) * hop / sr

    def crepe_pitch(start: float, end: float) -> float | None:
        mask = (times >= start) & (times < end)
        seg = f0[mask]
        seg = seg[np.isfinite(seg) & (seg > 0)]  # drop unvoiced (NaN) frames
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
    """Assign (string, fret) to each note via the shared polyphonic tab engine.

    Chord-aware DP minimising hand movement + fret span (see :mod:`tabs`);
    double-stops land on distinct strings. Notes outside the fretboard keep
    ``string``/``fret`` as ``None``.
    """
    from .tabs import assign_frets

    assign_frets(notes, config.tuning, config.frets)


def render_ascii_tab(
    notes: list[BassNote],
    config: BassTranscriptionConfig,
    columns: int = 80,
    title: str | None = None,
    tempo: float | None = None,
) -> str:
    """Render a chord-aware ASCII tab, wrapped into systems (see :mod:`tabs`)."""
    from .tabs import render_ascii_tab as _render

    return _render(notes, config.tuning, width=columns, title=title, tempo=tempo)
