"""Command-line interface.

    drum-extractor run song.mp3                 # full pipeline
    drum-extractor separate song.mp3            # stems only (Stage 1)
    drum-extractor transcribe-drums drums.wav   # Stage 2a
    drum-extractor transcribe-bass bass.wav     # Stage 2b
    drum-extractor notate transcription.json    # Stage 4

Run ``drum-extractor <command> -h`` for per-command options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import (
    BassTranscriptionConfig,
    DrumTranscriptionConfig,
    NotationConfig,
    PipelineConfig,
    QuantizeConfig,
    SeparationConfig,
)
from .logging_utils import configure_logging, get_logger

log = get_logger(__name__)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-o", "--output", default="output", help="Output directory (default: ./output)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose (debug) logging")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drum-extractor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"drum-extractor {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # run — full pipeline
    run = sub.add_parser("run", help="Full pipeline: song -> stems, MIDI, tab, sheet music")
    run.add_argument("audio", help="Input song (wav/mp3/flac/...)")
    _add_common(run)
    run.add_argument("--model", default="htdemucs_ft", help="Demucs model (default: htdemucs_ft)")
    run.add_argument("--device", default="auto", help="auto|cpu|cuda|mps (default: auto)")
    run.add_argument("--shifts", type=int, default=1, help="Demucs shifts (higher = slower, slightly cleaner)")
    run.add_argument("--grid", type=int, default=16, help="Quantization grid 1/N (16=sixteenths, 32 for double-kick)")
    run.add_argument("--drum-backend", default="adtof", choices=["adtof", "onset", "none"])
    run.add_argument("--no-bass", action="store_true", help="Skip bass transcription")
    run.add_argument("--no-quantize", action="store_true", help="Skip quantization")
    run.add_argument("--no-notation", action="store_true", help="Skip sheet-music generation")
    run.add_argument("--crepe", action="store_true", help="Octave-correct bass with torchcrepe")
    run.add_argument("--boost-double-kick", action="store_true", help="Recover fast double-kick (metal)")
    run.add_argument("--ensemble-model", default=None, help="audio-separator drum model to blend with Demucs")
    run.add_argument("--no-sonify", action="store_true", help="Skip the by-ear correction audio + onset CSV")

    # separate — Stage 1 only
    sep = sub.add_parser("separate", help="Stage 1 only: isolate stems with Demucs")
    sep.add_argument("audio", help="Input song")
    _add_common(sep)
    sep.add_argument("--model", default="htdemucs_ft")
    sep.add_argument("--device", default="auto")
    sep.add_argument("--shifts", type=int, default=1)
    sep.add_argument("--stems", default="drums,bass", help="Comma-separated stems to keep (default: drums,bass)")

    # transcribe-drums — Stage 2a
    td = sub.add_parser("transcribe-drums", help="Stage 2a: drum stem -> drum MIDI")
    td.add_argument("drum_stem", help="Isolated drum stem (wav/flac/mp3)")
    _add_common(td)
    td.add_argument("--backend", default="adtof", choices=["adtof", "onset", "none"])

    # transcribe-bass — Stage 2b
    tb = sub.add_parser("transcribe-bass", help="Stage 2b: bass stem -> bass MIDI + tab")
    tb.add_argument("bass_stem", help="Isolated bass stem")
    _add_common(tb)
    tb.add_argument("--crepe", action="store_true", help="Octave-correct with torchcrepe")

    # notate — Stage 4
    nt = sub.add_parser("notate", help="Stage 4: transcription.json or drum MIDI -> sheet music")
    nt.add_argument("source", help="transcription.json or a drum .mid file")
    _add_common(nt)
    nt.add_argument("--grid", type=int, default=16)
    nt.add_argument("--title", default="Drum Transcription")
    nt.add_argument("--no-pdf", action="store_true", help="Write MusicXML only, skip MuseScore PDF")

    return parser


def _cmd_run(args) -> int:
    config = PipelineConfig(
        output_dir=Path(args.output),
        separation=SeparationConfig(
            model=args.model, device=args.device, shifts=args.shifts, ensemble_drums_model=args.ensemble_model
        ),
        drums=DrumTranscriptionConfig(backend=args.drum_backend, boost_double_kick=args.boost_double_kick),
        bass=BassTranscriptionConfig(refine_with_crepe=args.crepe),
        quantize=QuantizeConfig(grid=args.grid),
        do_bass_transcription=not args.no_bass,
        do_quantize=not args.no_quantize,
        do_notation=not args.no_notation,
        do_sonify=not args.no_sonify,
    )
    from .pipeline import run_pipeline

    result = run_pipeline(args.audio, config)
    print(result.summary())
    return 0


def _cmd_separate(args) -> int:
    from .separation import separate

    config = SeparationConfig(
        model=args.model, device=args.device, shifts=args.shifts, stems=tuple(s.strip() for s in args.stems.split(","))
    )
    out_dir = Path(args.output) / Path(args.audio).stem / "stems"
    stems = separate(args.audio, out_dir, config)
    for name, path in stems.available().items():
        print(f"{name:>8}: {path}")
    return 0


def _cmd_transcribe_drums(args) -> int:
    from .drums import transcribe_drums
    from .midi_io import write_drum_midi

    hits = transcribe_drums(args.drum_stem, DrumTranscriptionConfig(backend=args.backend))
    out = Path(args.output) / f"{Path(args.drum_stem).stem}.mid"
    write_drum_midi(hits, out)
    print(f"{len(hits)} hits -> {out}")
    return 0


def _cmd_transcribe_bass(args) -> int:
    from .bass import render_ascii_tab, transcribe_bass
    from .midi_io import write_bass_midi

    config = BassTranscriptionConfig(refine_with_crepe=args.crepe)
    notes = transcribe_bass(args.bass_stem, config)
    stem = Path(args.bass_stem).stem
    out_mid = Path(args.output) / f"{stem}.mid"
    write_bass_midi(notes, out_mid)
    out_tab = Path(args.output) / f"{stem}.tab.txt"
    out_tab.parent.mkdir(parents=True, exist_ok=True)
    out_tab.write_text(render_ascii_tab(notes, config))
    print(f"{len(notes)} notes -> {out_mid}\ntab -> {out_tab}")
    return 0


def _cmd_notate(args) -> int:
    from .events import Transcription
    from .midi_io import read_drum_hits, read_drum_tempo
    from .notation import notate_drums

    source = Path(args.source)
    if source.suffix == ".json":
        transcription = Transcription.from_dict(json.loads(source.read_text()))
    else:  # treat as a drum MIDI — carry its embedded tempo so note positions are correct
        transcription = Transcription(drum_hits=read_drum_hits(source), tempo=read_drum_tempo(source))

    out_dir = Path(args.output) / source.stem
    results = notate_drums(
        transcription,
        out_dir,
        NotationConfig(title=args.title, render_pdf=not args.no_pdf),
        QuantizeConfig(grid=args.grid),
    )
    for name, path in results.items():
        print(f"{name:>10}: {path}")
    return 0


_COMMANDS = {
    "run": _cmd_run,
    "separate": _cmd_separate,
    "transcribe-drums": _cmd_transcribe_drums,
    "transcribe-bass": _cmd_transcribe_bass,
    "notate": _cmd_notate,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(getattr(args, "verbose", False))
    try:
        return _COMMANDS[args.command](args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # surface a clean message; --verbose for the traceback
        log.error("%s", exc)
        if getattr(args, "verbose", False):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
