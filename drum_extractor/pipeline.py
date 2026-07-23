"""End-to-end orchestration: song in, stems + MIDI + tab + sheet music out.

Each stage is independently toggleable and degrades gracefully: if an optional
stage's dependency is missing, that stage is skipped with a clear warning while
the rest of the pipeline still runs. Stage 1 (separation) is the always-works
core; everything else builds on the stems it produces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import PipelineConfig
from .errors import DrumExtractorError, MissingDependencyError
from .events import Stems, Transcription
from .logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class PipelineResult:
    """Paths and data produced by a pipeline run."""

    stems: Stems = field(default_factory=Stems)
    transcription: Transcription = field(default_factory=Transcription)
    drum_midi: Path | None = None
    bass_midi: Path | None = None
    bass_tab: Path | None = None
    musicxml: Path | None = None
    pdf: Path | None = None
    drum_sonification: Path | None = None
    onset_csv: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Pipeline results:"]
        for label, value in [
            ("drums stem", self.stems.drums),
            ("bass stem", self.stems.bass),
            ("drum MIDI", self.drum_midi),
            ("bass MIDI", self.bass_midi),
            ("bass tab", self.bass_tab),
            ("MusicXML", self.musicxml),
            ("drum sheet PDF", self.pdf),
            ("drum sonified", self.drum_sonification),
            ("onset CSV", self.onset_csv),
        ]:
            if value:
                lines.append(f"  - {label:<14} {value}")
        if self.warnings:
            lines.append("Skipped / warnings:")
            lines.extend(f"  ! {w}" for w in self.warnings)
        return "\n".join(lines)


def run_pipeline(
    audio_path: str | Path,
    config: PipelineConfig | None = None,
    on_stage=None,
) -> PipelineResult:
    """Run the configured stages on ``audio_path`` and return a result bundle.

    ``on_stage`` (optional) is called with a short stage name as each stage
    begins — used by UIs to show progress. Exceptions it raises are swallowed.
    """
    config = config or PipelineConfig()
    audio_path = Path(audio_path)
    song_dir = config.output_dir / audio_path.stem
    song_dir.mkdir(parents=True, exist_ok=True)
    result = PipelineResult()

    def notify(stage: str) -> None:
        if on_stage:
            try:
                on_stage(stage)
            except Exception:  # a UI callback must never break the pipeline
                pass

    # --- Stage 1: separation (the reliable core) ---
    if config.do_separation:
        from .separation import separate

        notify("separating")
        result.stems = separate(audio_path, song_dir / "stems", config.separation)
    else:
        # Allow re-running later stages on pre-existing stems.
        result.stems = _discover_existing_stems(song_dir / "stems")

    # --- Phase 4: optional ensemble drum stem (cuts distorted-guitar bleed) ---
    if config.separation.ensemble_drums_model and result.stems.drums:
        _try(result, "ensemble drums", lambda: _do_ensemble(result, config, audio_path, song_dir))

    # --- Stage 2a: drum transcription ---
    if config.do_drum_transcription and result.stems.drums:
        notify("transcribing drums")
        _try(result, "drum transcription", lambda: _do_drums(result, config, song_dir))
    elif config.do_drum_transcription:
        result.warnings.append("drum transcription: no drum stem available")

    # --- Stage 2b: bass transcription ---
    if config.do_bass_transcription and result.stems.bass:
        notify("transcribing bass")
        _try(result, "bass transcription", lambda: _do_bass(result, config, song_dir))
    elif config.do_bass_transcription:
        result.warnings.append("bass transcription: no bass stem available")

    # --- Stage 3: quantization (needs drum hits + an audio reference) ---
    if config.do_quantize and result.transcription.drum_hits:
        notify("quantizing")
        ref = result.stems.drums or audio_path
        _try(result, "quantization", lambda: _do_quantize(result, config, ref))

    # Persist the transcription IR NOW — before the optional, more fragile
    # notation/sonify stages — so a late-stage failure can't discard the
    # expensive separation + transcription + quantization work (the IR can be
    # re-notated later via `drum-extractor notate transcription.json`).
    _write_transcription_json(result.transcription, song_dir / "transcription.json")

    # --- Stage 4: notation ---
    if config.do_notation and result.transcription.drum_hits:
        notify("engraving sheet music")
        _try(result, "notation", lambda: _do_notation(result, config, song_dir))

    # --- Phase 4: correction aids (sonification + onset CSV) ---
    if config.do_sonify and result.transcription.drum_hits:
        notify("rendering correction aids")
        _try(result, "sonification", lambda: _do_sonify(result, song_dir))

    log.info("\n%s", result.summary())
    return result


def _try(result: PipelineResult, label: str, fn) -> None:
    """Run an optional stage, converting any failure into a recorded warning.

    Optional stages must degrade gracefully: a missing dependency, a
    DrumExtractorError, or an unexpected runtime error in one stage is logged
    and recorded but does not abort the run or prevent later independent stages.
    (Stage 1 separation is deliberately NOT wrapped — it is the core and its
    failures should surface.)
    """
    try:
        fn()
    except (MissingDependencyError, DrumExtractorError) as exc:
        msg = f"{label}: {exc}"
        log.warning(msg)
        result.warnings.append(msg)
    except Exception as exc:  # keep the pipeline's graceful-degradation contract
        msg = f"{label}: unexpected {type(exc).__name__}: {exc}"
        log.warning(msg)
        log.debug("%s failed", label, exc_info=True)  # full traceback under -v
        result.warnings.append(msg)


def _do_drums(result: PipelineResult, config: PipelineConfig, song_dir: Path) -> None:
    from .drums import transcribe_drums
    from .midi_io import write_drum_midi

    hits = transcribe_drums(result.stems.drums, config.drums)
    result.transcription.drum_hits = hits
    if hits:
        result.drum_midi = write_drum_midi(hits, song_dir / "drums.mid", result.transcription.tempo)


def _do_bass(result: PipelineResult, config: PipelineConfig, song_dir: Path) -> None:
    from .bass import render_ascii_tab, transcribe_bass
    from .midi_io import write_bass_midi

    notes = transcribe_bass(result.stems.bass, config.bass)
    result.transcription.bass_notes = notes
    if notes:
        result.bass_midi = write_bass_midi(notes, song_dir / "bass.mid", result.transcription.tempo)
        tab_path = song_dir / "bass.tab.txt"
        tab_path.write_text(render_ascii_tab(notes, config.bass))
        result.bass_tab = tab_path


def _do_quantize(result: PipelineResult, config: PipelineConfig, ref_audio: Path) -> None:
    from .quantize import quantize

    quantize(result.transcription, ref_audio, config.quantize)
    # Rewrite drum MIDI with quantized timing.
    if result.drum_midi and result.transcription.drum_hits:
        from .midi_io import write_drum_midi

        write_drum_midi(result.transcription.drum_hits, result.drum_midi, result.transcription.tempo)


def _do_notation(result: PipelineResult, config: PipelineConfig, song_dir: Path) -> None:
    from .notation import notate_drums

    out = notate_drums(result.transcription, song_dir, config.notation, config.quantize)
    result.musicxml = out.get("musicxml")
    result.pdf = out.get("pdf")


def _do_ensemble(result: PipelineResult, config: PipelineConfig, audio_path: Path, song_dir: Path) -> None:
    from .ensemble import ensemble_drums

    result.stems.drums = ensemble_drums(
        audio_path,
        result.stems.drums,
        song_dir / "stems",
        config.separation.ensemble_drums_model,
        algorithm=config.separation.ensemble_algorithm,
    )


def _do_sonify(result: PipelineResult, song_dir: Path) -> None:
    from .sonify import sonify_drums, write_onset_csv

    result.drum_sonification = sonify_drums(result.transcription.drum_hits, song_dir / "drums_sonified.wav")
    result.onset_csv = write_onset_csv(result.transcription.drum_hits, song_dir / "drum_onsets.csv")


def _discover_existing_stems(stems_dir: Path) -> Stems:
    stems = Stems()
    if not stems_dir.exists():
        return stems
    for name in ("drums", "bass", "other", "vocals", "guitar", "piano"):
        for ext in ("wav", "flac", "mp3"):
            candidate = stems_dir / f"{name}.{ext}"
            if candidate.exists():
                setattr(stems, name, candidate)
                break
    return stems


def _write_transcription_json(transcription: Transcription, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transcription.to_dict(), indent=2))
