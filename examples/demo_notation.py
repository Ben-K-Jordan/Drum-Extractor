#!/usr/bin/env python3
"""Generate a drum chart from a synthetic groove — no audio or models needed.

Useful for verifying the notation stage (Stage 4) end-to-end on any machine:

    pip install -e ".[notation]"
    python examples/demo_notation.py

Writes ``examples/out/drums.musicxml`` (open it in MuseScore/any notation
editor). If the MuseScore CLI is on PATH it also renders a PDF.
"""

from __future__ import annotations

from pathlib import Path

from drum_extractor.config import NotationConfig, QuantizeConfig
from drum_extractor.events import DrumHit, Transcription
from drum_extractor.notation import notate_drums


def basic_rock_beat(bars: int = 2, bpm: float = 120.0) -> Transcription:
    """A simple rock groove: 8th-note hi-hats, kick on 1 & 3, snare on 2 & 4."""
    hits: list[DrumHit] = []
    beats_per_bar = 4
    sec_per_beat = 60.0 / bpm
    for bar in range(1, bars + 1):
        base = (bar - 1) * beats_per_bar * sec_per_beat
        for i in range(8):  # eighth-note hats
            hits.append(DrumHit(time=base + i * sec_per_beat / 2, instrument="hihat_closed", bar=bar, beat=i * 0.5))
        hits.append(DrumHit(time=base + 0 * sec_per_beat, instrument="kick", bar=bar, beat=0.0))
        hits.append(DrumHit(time=base + 2 * sec_per_beat, instrument="kick", bar=bar, beat=2.0))
        hits.append(DrumHit(time=base + 1 * sec_per_beat, instrument="snare", bar=bar, beat=1.0))
        hits.append(DrumHit(time=base + 3 * sec_per_beat, instrument="snare", bar=bar, beat=3.0))
    hits.append(DrumHit(time=0.0, instrument="crash", bar=1, beat=0.0))  # crash on the downbeat
    return Transcription(drum_hits=hits, tempo=bpm, time_signature=(4, 4))


def main() -> None:
    out_dir = Path(__file__).parent / "out"
    transcription = basic_rock_beat()
    results = notate_drums(
        transcription,
        out_dir,
        NotationConfig(title="Demo Rock Groove"),
        QuantizeConfig(grid=16),
    )
    for name, path in results.items():
        print(f"{name:>10}: {path}")
    print("\nOpen the .musicxml in MuseScore (or run the full pipeline on a real song).")


if __name__ == "__main__":
    main()
