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
    guitar_midi: Path | None = None
    guitar_tab: Path | None = None
    bass_gp: Path | None = None
    guitar_gp: Path | None = None
    musicxml: Path | None = None
    pdf: Path | None = None
    drum_sonification: Path | None = None
    onset_csv: Path | None = None
    transcription_json: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Pipeline results:"]
        for label, value in [
            ("drums stem", self.stems.drums),
            ("bass stem", self.stems.bass),
            ("guitar stem", self.stems.guitar),
            ("drum MIDI", self.drum_midi),
            ("bass MIDI", self.bass_midi),
            ("bass tab", self.bass_tab),
            ("guitar MIDI", self.guitar_midi),
            ("guitar tab", self.guitar_tab),
            ("bass GP5", self.bass_gp),
            ("guitar GP5", self.guitar_gp),
            ("MusicXML", self.musicxml),
            ("drum sheet PDF", self.pdf),
            ("drum sonified", self.drum_sonification),
            ("onset CSV", self.onset_csv),
            ("transcription", self.transcription_json),
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
    if config.do_separation and not audio_path.exists():
        from .errors import AudioLoadError

        # Validate BEFORE mkdir so a typo'd filename doesn't litter the output
        # tree with an empty directory.
        raise AudioLoadError(f"Input audio not found: {audio_path}")
    song_title = audio_path.stem
    song_dir = config.output_dir / song_title
    song_dir.mkdir(parents=True, exist_ok=True)
    result = PipelineResult()

    # Sheet music titled after the song, unless the caller set an explicit title.
    from .config import NotationConfig as _NC

    # Effective notation config, computed WITHOUT mutating the caller's
    # config: a PipelineConfig reused for a second song must not carry the
    # first song's title into the second song's sheet.
    import dataclasses as _dc

    notation_config = config.notation
    if notation_config.title == _NC.title:
        notation_config = _dc.replace(notation_config, title=song_title)

    # A rerun with a stage now disabled must not leave that stage's previous
    # outputs lying around contradicting the fresh transcription.json.
    if not config.do_bass_transcription:
        _remove_stale(song_dir, "bass.mid", "bass.tab.txt", "bass.gp5")
    if not config.do_guitar_transcription:
        _remove_stale(song_dir, "guitar.mid", "guitar.tab.txt", "guitar.gp5")
    if not config.do_drum_transcription:
        _remove_stale(song_dir, "drums.mid")
    if not config.do_notation:
        _remove_stale(song_dir, "drums.musicxml", "drums.pdf")
    if not config.do_sonify:
        _remove_stale(song_dir, "drums_sonified.wav", "drum_onsets.csv")

    def notify(stage: str) -> None:
        if on_stage:
            try:
                on_stage(stage)
            except Exception:  # a UI callback must never break the pipeline
                pass

    # Guitar tabs need a guitar stem, which only the 6-stem model produces.
    # Auto-upgrade rather than silently skipping the stage the user asked for.
    if config.do_guitar_transcription and "guitar" not in config.separation.stems:
        config.separation.stems = tuple(config.separation.stems) + ("guitar",)
        if config.separation.model in ("htdemucs", "htdemucs_ft"):
            log.info(
                "Guitar transcription enabled: switching Demucs model to htdemucs_6s "
                "(the only one with a guitar stem; slightly more bleed on dense mixes)."
            )
            config.separation.model = "htdemucs_6s"

    # --- Stage 1: separation (the reliable core) ---
    if config.do_separation:
        from .separation import separate

        notify("separating")
        result.stems = separate(
            audio_path,
            song_dir / "stems",
            config.separation,
            progress=lambda f: notify(f"separating — {int(f * 100)}%"),
        )
    else:
        # Allow re-running later stages on pre-existing stems.
        result.stems = _discover_existing_stems(song_dir / "stems")

    # --- Phase 4: optional ensemble drum stem (cuts distorted-guitar bleed) ---
    if config.separation.ensemble_drums_model and result.stems.drums:
        notify("ensembling drums")
        _try(result, "ensemble drums", lambda: _do_ensemble(result, config, audio_path, song_dir))

    # --- Stage 2a: drum transcription ---
    if config.do_drum_transcription and result.stems.drums:
        notify("transcribing drums")
        _try(result, "drum transcription", lambda: _do_drums(result, config, song_dir))
    elif config.do_drum_transcription:
        result.warnings.append("drum transcription: no drum stem available")

    # Track which string stages RAN SUCCESSFULLY: stale-output removal for an
    # empty result must never fire on a stage that errored — a transient
    # failure must not destroy the previous run's good files.
    stage_ok = {"bass": False, "guitar": False}

    # --- Stage 2b: bass transcription ---
    if config.do_bass_transcription and result.stems.bass:
        notify("transcribing bass")

        def _bass_stage() -> None:
            _do_bass(result, config, song_dir)
            stage_ok["bass"] = True

        _try(result, "bass transcription", _bass_stage)
    elif config.do_bass_transcription:
        result.warnings.append("bass transcription: no bass stem available")

    # --- Optional: guitar transcription (needs the 6-stem model's guitar stem) ---
    if config.do_guitar_transcription and result.stems.guitar:
        notify("transcribing guitar")

        def _guitar_stage() -> None:
            _do_guitar(result, config, song_dir)
            stage_ok["guitar"] = True

        _try(result, "guitar transcription", _guitar_stage)
    elif config.do_guitar_transcription:
        result.warnings.append("guitar transcription: no guitar stem available (needs htdemucs_6s)")

    # --- Stage 3: quantization (needs drum hits + an audio reference) ---
    if config.do_quantize and result.transcription.drum_hits:
        notify("quantizing")
        ref = result.stems.drums or audio_path
        _try(result, "quantization", lambda: _do_quantize(result, config, ref))

    # Persist the transcription IR NOW — before the optional, more fragile
    # notation/sonify stages — so a late-stage failure can't discard the
    # expensive separation + transcription + quantization work (the IR can be
    # re-notated later via `drum-extractor notate transcription.json`).
    result.transcription_json = _write_transcription_json(
        result.transcription, song_dir / "transcription.json"
    )

    # --- Tabs + Guitar Pro files, AFTER quantization so they carry the real
    # tempo: written earlier they'd be built on the 120 BPM default grid.
    # Runs even with zero notes so a successful-but-empty rerun cleans up
    # now-contradicting files from a previous run. ---
    if result.transcription.bass_notes or result.transcription.guitar_notes:
        notify("writing tabs")
    _try(result, "tab rendering",
         lambda: _do_string_outputs(result, config, song_dir, song_title, stage_ok))

    # --- Stage 4: notation ---
    if config.do_notation and result.transcription.drum_hits:
        notify("engraving sheet music")
        _try(result, "notation", lambda: _do_notation(result, config, song_dir, notation_config))

    # --- Phase 4: correction aids (sonification + onset CSV) ---
    if config.do_sonify and result.transcription.drum_hits:
        notify("rendering correction aids")
        _try(result, "sonification", lambda: _do_sonify(result, song_dir))

    # debug, not info: the CLI prints summary() to stdout already, and a
    # terminal user would otherwise see the whole block twice back-to-back.
    log.debug("\n%s", result.summary())
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
    from .bass import transcribe_bass

    # Tab/GP5/MIDI files are written after quantization (see
    # _do_string_outputs) so they carry the song's real tempo.
    result.transcription.bass_notes = transcribe_bass(result.stems.bass, config.bass)


def _do_guitar(result: PipelineResult, config: PipelineConfig, song_dir: Path) -> None:
    from .guitar import transcribe_guitar

    result.transcription.guitar_notes = transcribe_guitar(result.stems.guitar, config.guitar)


def _do_string_outputs(result: PipelineResult, config: PipelineConfig, song_dir: Path,
                       title: str, stage_ok: dict[str, bool]) -> None:
    """Write bass/guitar MIDI + ASCII tab + .gp5, tempo-aware.

    Stale files from a previous run are removed only when the stage RAN
    SUCCESSFULLY and legitimately produced nothing — an errored stage must
    leave the previous outputs alone.
    """
    from .midi_io import write_bass_midi
    from .tabs import render_ascii_tab

    tempo = result.transcription.tempo
    bpb = config.quantize.beats_per_bar

    bass_notes = result.transcription.bass_notes
    if bass_notes:
        result.bass_midi = write_bass_midi(bass_notes, song_dir / "bass.mid", tempo)
        tab_path = song_dir / "bass.tab.txt"
        tab_path.write_text(render_ascii_tab(
            bass_notes, config.bass.tuning, title=f"{title} — bass", tempo=tempo) + "\n")
        result.bass_tab = tab_path
        result.bass_gp = _maybe_gp(bass_notes, config.bass.tuning, song_dir / "bass.gp5",
                                   tempo, title, "Bass", 33, bpb)
    elif stage_ok["bass"]:
        _remove_stale(song_dir, "bass.mid", "bass.tab.txt", "bass.gp5")

    guitar_notes = result.transcription.guitar_notes
    if guitar_notes:
        result.guitar_midi = write_bass_midi(
            guitar_notes, song_dir / "guitar.mid", tempo, program=30, name="Guitar"
        )
        tab_path = song_dir / "guitar.tab.txt"
        tab_path.write_text(render_ascii_tab(
            guitar_notes, config.guitar.tuning, title=f"{title} — guitar", tempo=tempo) + "\n")
        result.guitar_tab = tab_path
        result.guitar_gp = _maybe_gp(guitar_notes, config.guitar.tuning, song_dir / "guitar.gp5",
                                     tempo, title, "Guitar", 30, bpb)
    elif stage_ok["guitar"]:
        _remove_stale(song_dir, "guitar.mid", "guitar.tab.txt", "guitar.gp5")


def _maybe_gp(notes, tuning, path: Path, tempo, title: str, track_name: str,
              instrument: int, beats_per_bar: int = 4) -> Path | None:
    """Write a .gp5 if PyGuitarPro is installed; quietly skip otherwise."""
    from .gp_export import gp_available, write_gp5

    if not gp_available():
        log.debug("PyGuitarPro not installed; skipping %s", path.name)
        return None
    try:
        return write_gp5(notes, tuning, path, tempo=tempo, title=title,
                         track_name=track_name, instrument=instrument,
                         artist="drum-extractor transcription", beats_per_bar=beats_per_bar)
    except Exception as exc:  # optional nicety; never fail the stage over it
        log.warning("Guitar Pro export skipped (%s)", exc)
        return None


def _remove_stale(song_dir: Path, *names: str) -> None:
    for name in names:
        p = song_dir / name
        if p.exists():
            log.info("Removing stale output from a previous run: %s", p.name)
            p.unlink(missing_ok=True)


def _do_quantize(result: PipelineResult, config: PipelineConfig, ref_audio: Path) -> None:
    from .quantize import quantize

    quantize(result.transcription, ref_audio, config.quantize)
    # Rewrite drum MIDI with quantized timing.
    if result.drum_midi and result.transcription.drum_hits:
        from .midi_io import write_drum_midi

        write_drum_midi(result.transcription.drum_hits, result.drum_midi, result.transcription.tempo)


def _do_notation(result: PipelineResult, config: PipelineConfig, song_dir: Path, notation_config) -> None:
    from .notation import notate_drums

    out = notate_drums(result.transcription, song_dir, notation_config, config.quantize)
    result.musicxml = out.get("musicxml")
    result.pdf = out.get("pdf")
    if notation_config.render_pdf and result.pdf is None:
        # Otherwise the skip is only a mid-run stderr line, invisible in the
        # final summary.
        result.warnings.append(
            "notation: PDF skipped — MuseScore not found (the MusicXML was written; "
            "install MuseScore 4 for PDF export)"
        )


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


def _write_transcription_json(transcription: Transcription, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transcription.to_dict(), indent=2))
    return path
