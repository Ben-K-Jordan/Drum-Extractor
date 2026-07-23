"""Configuration objects for each stage and the overall pipeline.

Defaults are chosen for the target use case (metal/punk/rock, learner-oriented):
the fine-tuned Demucs model, a 1/16 quantization grid, transcription driven from
the isolated drum stem. Override any field from the CLI or when calling the API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SeparationConfig:
    """Stage 1 — Demucs source separation."""

    model: str = "htdemucs_ft"  # fine-tuned: best drums+bass at ~4x the compute
    device: str = "auto"  # "auto" -> cuda if available else cpu
    shifts: int = 1  # >1 averages time-shifted passes for a small quality gain
    overlap: float = 0.25
    stems: tuple[str, ...] = ("drums", "bass")  # which stems to keep
    mp3: bool = False  # write mp3 instead of wav
    mp3_bitrate: int = 320
    jobs: int = 0  # parallel jobs (0 = auto)
    # Phase 4: blend a second drums model (an audio-separator model filename,
    # e.g. a RoFormer/SCNet drum checkpoint) with Demucs to cut guitar bleed on
    # dense metal. None = Demucs only. Requires `pip install audio-separator`.
    ensemble_drums_model: str | None = None


@dataclass
class DrumTranscriptionConfig:
    """Stage 2a — drum audio -> drum hits."""

    backend: str = "adtof"  # "adtof" | "onset" (librosa fallback) | "none"
    # Command to invoke ADTOF. Default targets ADTOF-pytorch, whose `adtof`
    # console script takes `--audio`/`--out` and writes MIDI. There is NO
    # `python -m adtof`. Override for a different install (e.g. omnizart:
    # ("omnizart","drum","transcribe","{input}","-o","{output}")). ``{input}``
    # and ``{output}`` are substituted at call time.
    adtof_command: tuple[str, ...] = ("adtof", "--audio", "{input}", "--out", "{output}")
    default_velocity: int = 96
    # Phase 4: recover fast double-bass the full-kit model merges, via a
    # low-pass kick-band onset detector. Targets metal's biggest weak spot.
    boost_double_kick: bool = False


@dataclass
class BassTranscriptionConfig:
    """Stage 2b — bass audio -> bass notes."""

    backend: str = "basic_pitch"  # "basic_pitch" | "none"
    min_frequency: float = 32.7  # C1; standard 4-string low E is 41 Hz
    max_frequency: float = 400.0
    # basic-pitch note-gating knobs. Its 127.7 ms default drops fast 16ths/ghost
    # notes; lower it for busy metal bass. Thresholds default to basic-pitch's.
    minimum_note_length_ms: float = 90.0
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    refine_with_crepe: bool = False  # octave-correct with torchcrepe (needs the extra)
    # Standard 4-string bass tuning (E1 A1 D2 G2) as MIDI note numbers, low->high.
    tuning: tuple[int, ...] = (28, 33, 38, 43)
    frets: int = 24


@dataclass
class QuantizeConfig:
    """Stage 3 — snap onsets to a musical grid."""

    backend: str = "madmom"  # "madmom" | "librosa" | "none"
    grid: int = 16  # 1/grid note resolution: 16 = sixteenths, 32 for fast double-kick
    beats_per_bar: int = 4
    beat_unit: int = 4
    fixed_tempo: float | None = None  # skip tempo detection if you already know it


@dataclass
class NotationConfig:
    """Stage 4 — MIDI/IR -> engraved drum sheet music."""

    backend: str = "music21"  # builds MusicXML; "midi_import" hands the MIDI to MuseScore
    musescore_command: str = "mscore"  # or "musescore4", "MuseScore4.exe", full path
    render_pdf: bool = True  # requires MuseScore on PATH
    title: str = "Drum Transcription"


@dataclass
class PipelineConfig:
    """Top-level config aggregating every stage."""

    output_dir: Path = Path("output")
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    drums: DrumTranscriptionConfig = field(default_factory=DrumTranscriptionConfig)
    bass: BassTranscriptionConfig = field(default_factory=BassTranscriptionConfig)
    quantize: QuantizeConfig = field(default_factory=QuantizeConfig)
    notation: NotationConfig = field(default_factory=NotationConfig)

    # Which stages to run. Stage 1 is the always-works core; the rest are opt-in.
    do_separation: bool = True
    do_drum_transcription: bool = True
    do_bass_transcription: bool = True
    do_quantize: bool = True
    do_notation: bool = True
    # Phase 4 correction aid: render the drum transcription back to audio (+ an
    # onset CSV) so you can A/B it against the stem by ear.
    do_sonify: bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
