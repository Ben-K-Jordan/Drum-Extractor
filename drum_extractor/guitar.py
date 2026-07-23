"""Optional stage — guitar transcription (guitar stem -> notes + chord tab).

Reuses the bass path's basic-pitch engine (it's polyphonic, so chords come
through) and the shared chord-aware tab assigner. Requires a guitar stem,
which only the 6-stem Demucs model produces; the pipeline auto-switches when
this stage is enabled.

Honesty note: high-gain rhythm guitar is the hardest transcription target in
this whole project — harmonics masquerade as extra notes and palm-muted chugs
blur together. Expect clean/lead parts to be genuinely useful and dense
distorted riffs to be a sketch you correct by ear.
"""

from __future__ import annotations

from pathlib import Path

from .config import GuitarTranscriptionConfig
from .events import BassNote
from .logging_utils import get_logger

log = get_logger(__name__)


def transcribe_guitar(guitar_stem: str | Path, config: GuitarTranscriptionConfig | None = None) -> list[BassNote]:
    """Transcribe an isolated guitar stem into notes with string/fret assigned."""
    config = config or GuitarTranscriptionConfig()
    if config.backend == "none":
        return []
    if config.backend != "basic_pitch":
        raise ValueError(f"Unknown guitar transcription backend: {config.backend!r}")

    from .bass import basic_pitch_notes
    from .tabs import assign_frets

    notes = basic_pitch_notes(
        guitar_stem,
        feature="Guitar transcription",
        min_frequency=config.min_frequency,
        max_frequency=config.max_frequency,
        minimum_note_length_ms=config.minimum_note_length_ms,
        onset_threshold=config.onset_threshold,
        frame_threshold=config.frame_threshold,
    )
    assign_frets(notes, config.tuning, config.frets)
    log.info("Guitar: %d notes transcribed", len(notes))
    return notes


def render_guitar_tab(notes: list[BassNote], config: GuitarTranscriptionConfig | None = None, width: int = 76) -> str:
    """Chord-aware ASCII guitar tab (see :mod:`tabs`)."""
    from .tabs import render_ascii_tab

    config = config or GuitarTranscriptionConfig()
    return render_ascii_tab(notes, config.tuning, width=width)
