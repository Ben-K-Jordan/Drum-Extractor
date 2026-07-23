"""drum_extractor — isolate drums & bass from a song and transcribe drums to sheet music.

Four-stage pipeline for metal/punk/rock:
    1. Separation      Demucs v4        full mix  -> drums.wav + bass.wav
    2a. Drum ADT       ADTOF            drums.wav -> drum MIDI
    2b. Bass transcr.  basic-pitch      bass.wav  -> bass MIDI + tab
    3. Quantization    madmom           snap onsets to a bar grid
    4. Notation        music21+MuseScore-> drum sheet music (MusicXML / PDF)

Only pure-Python core objects are imported eagerly here. Heavy dependencies
(torch, demucs, librosa, music21, ...) load lazily inside each stage, so
``import drum_extractor`` works with nothing but the standard library installed.
"""

from __future__ import annotations

from .config import (
    BassTranscriptionConfig,
    DrumTranscriptionConfig,
    NotationConfig,
    PipelineConfig,
    QuantizeConfig,
    SeparationConfig,
)
from .errors import (
    AudioLoadError,
    DrumExtractorError,
    ExternalToolError,
    MissingDependencyError,
)
from .events import BassNote, DrumHit, Stems, Transcription
from .pipeline import PipelineResult, run_pipeline

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "run_pipeline",
    "PipelineResult",
    "PipelineConfig",
    "SeparationConfig",
    "DrumTranscriptionConfig",
    "BassTranscriptionConfig",
    "QuantizeConfig",
    "NotationConfig",
    "DrumHit",
    "BassNote",
    "Stems",
    "Transcription",
    "DrumExtractorError",
    "MissingDependencyError",
    "ExternalToolError",
    "AudioLoadError",
]
