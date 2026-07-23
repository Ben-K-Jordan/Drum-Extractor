"""MIDI reading/writing for drums and bass, via pretty_midi.

Isolated here so the rest of the codebase deals in :mod:`drum_extractor.events`
dataclasses and never touches MIDI internals directly.
"""

from __future__ import annotations

from pathlib import Path

from .errors import MissingDependencyError
from .events import BassNote, DrumHit
from .gm_drum_map import canonical_to_gm, gm_to_canonical
from .logging_utils import get_logger

log = get_logger(__name__)


def _pretty_midi():
    try:
        import pretty_midi  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        # pretty_midi is a core dependency, not an extra, so point at the package
        # directly (there is no `drum-extractor[midi]` extra).
        raise MissingDependencyError("MIDI export/import", "pretty_midi") from exc
    return pretty_midi


def write_drum_midi(hits: list[DrumHit], path: Path, tempo: float | None = None, hit_seconds: float = 0.05) -> Path:
    """Write drum hits to a channel-10 (is_drum) MIDI file.

    Drum hits are instantaneous, but MIDI notes need a duration; ``hit_seconds``
    gives each a short fixed length purely so the note is valid. Notation cares
    only about onsets, not these durations.
    """
    pm = _pretty_midi()
    midi = pm.PrettyMIDI(initial_tempo=float(tempo) if tempo else 120.0)
    drum = pm.Instrument(program=0, is_drum=True, name="Drums")
    for h in hits:
        note = pm.Note(
            velocity=int(max(1, min(127, h.velocity))),
            pitch=canonical_to_gm(h.instrument),
            start=float(h.time),
            end=float(h.time) + hit_seconds,
        )
        drum.notes.append(note)
    midi.instruments.append(drum)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    log.info("Wrote drum MIDI: %s (%d hits)", path, len(hits))
    return path


def write_bass_midi(notes: list[BassNote], path: Path, tempo: float | None = None) -> Path:
    """Write bass notes to a standard (pitched) MIDI file, program 33 (Electric Bass)."""
    pm = _pretty_midi()
    midi = pm.PrettyMIDI(initial_tempo=float(tempo) if tempo else 120.0)
    bass = pm.Instrument(program=33, is_drum=False, name="Bass")
    for n in notes:
        end = max(float(n.end), float(n.start) + 0.03)
        bass.notes.append(
            pm.Note(velocity=int(max(1, min(127, n.velocity))), pitch=int(n.pitch), start=float(n.start), end=end)
        )
    midi.instruments.append(bass)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    log.info("Wrote bass MIDI: %s (%d notes)", path, len(notes))
    return path


def read_drum_hits(path: Path, default_velocity: int = 96) -> list[DrumHit]:
    """Read a drum MIDI file back into :class:`DrumHit` objects.

    Reads every drum-channel instrument; non-drum tracks are ignored.
    """
    pm = _pretty_midi()
    midi = pm.PrettyMIDI(str(path))
    hits: list[DrumHit] = []
    for inst in midi.instruments:
        if not inst.is_drum:
            continue
        for note in inst.notes:
            hits.append(
                DrumHit(
                    time=float(note.start),
                    instrument=gm_to_canonical(note.pitch),
                    velocity=int(note.velocity) or default_velocity,
                )
            )
    hits.sort(key=lambda h: h.time)
    return hits


def read_drum_tempo(path: Path) -> float | None:
    """Read the first tempo (BPM) embedded in a MIDI file, if any.

    Used so ``notate <file>.mid`` engraves at the file's real tempo instead of
    defaulting to 120 BPM (which scales every note's position for other tempos).
    """
    pm = _pretty_midi()
    midi = pm.PrettyMIDI(str(path))
    _times, tempi = midi.get_tempo_changes()
    if len(tempi):
        return float(tempi[0])
    try:
        return float(midi.estimate_tempo())
    except Exception:  # pragma: no cover - estimate_tempo needs >=2 notes
        return None


def read_bass_notes(path: Path) -> list[BassNote]:
    """Read a pitched MIDI file into :class:`BassNote` objects (all non-drum tracks)."""
    pm = _pretty_midi()
    midi = pm.PrettyMIDI(str(path))
    notes: list[BassNote] = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            notes.append(
                BassNote(start=float(note.start), end=float(note.end), pitch=int(note.pitch), velocity=int(note.velocity))
            )
    notes.sort(key=lambda n: n.start)
    return notes
