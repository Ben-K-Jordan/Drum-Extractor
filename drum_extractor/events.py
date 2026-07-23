"""Intermediate representation passed between pipeline stages.

Keeping a small, plain-dataclass IR (rather than threading MIDI objects around)
means each stage has one clear contract: separation -> stems, transcription ->
``DrumHit``/``BassNote`` lists, quantization annotates timing, notation renders.
Everything is JSON-serialisable via :meth:`Transcription.to_dict` so intermediate
results can be cached or inspected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DrumHit:
    """A single drum onset.

    ``time`` is seconds from the start of the audio. After quantization it is
    snapped to the grid and ``bar``/``beat`` are filled in (1-indexed bar, and
    beat position within the bar in BEAT UNITS of the time signature — e.g.
    0..5 in 6/8 — not quarter notes; notation scales by ``4/beat_unit``).
    """

    time: float
    instrument: str  # canonical name from gm_drum_map.CANONICAL_INSTRUMENTS
    velocity: int = 96
    bar: int | None = None
    beat: float | None = None


@dataclass
class BassNote:
    """A single bass note. ``pitch`` is a MIDI note number."""

    start: float
    end: float
    pitch: int
    velocity: int = 80
    # Filled in by the tab mapper: string index (0 = lowest string) + fret.
    string: int | None = None
    fret: int | None = None


@dataclass
class Stems:
    """Paths to the separated audio produced by Stage 1."""

    drums: Path | None = None
    bass: Path | None = None
    other: Path | None = None
    vocals: Path | None = None
    guitar: Path | None = None
    piano: Path | None = None

    def available(self) -> dict[str, Path]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Transcription:
    """Everything the pipeline knows about a song after transcription."""

    drum_hits: list[DrumHit] = field(default_factory=list)
    bass_notes: list[BassNote] = field(default_factory=list)
    tempo: float | None = None
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    time_signature: tuple[int, int] = (4, 4)

    def to_dict(self) -> dict:
        return {
            "drum_hits": [asdict(h) for h in self.drum_hits],
            "bass_notes": [asdict(n) for n in self.bass_notes],
            "tempo": self.tempo,
            "beats": self.beats,
            "downbeats": self.downbeats,
            "time_signature": list(self.time_signature),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Transcription:
        return cls(
            drum_hits=[DrumHit(**h) for h in d.get("drum_hits", [])],
            bass_notes=[BassNote(**n) for n in d.get("bass_notes", [])],
            tempo=d.get("tempo"),
            beats=list(d.get("beats", [])),
            downbeats=list(d.get("downbeats", [])),
            time_signature=tuple(d.get("time_signature", (4, 4))),  # type: ignore[arg-type]
        )
