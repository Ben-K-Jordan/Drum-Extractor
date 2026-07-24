"""Build the demo song used in the README — no copyrighted audio involved.

Synthesizes a rendition of the classic Smells-Like-Teen-Spirit-style
four-power-chord riff (F5 Bb5 Ab5 Db5, ~117 BPM) as separated stems using the
repo's own drum synthesizer plus simple bass/guitar synths, then runs the REAL
pipeline on it (basic-pitch bass transcription, onset drum backend,
quantization, sheet engraving, .gp5 export) — everything except Demucs, which
is pointless on stems we already have.

    pip install -e ".[drums,bass,notation,gp]"
    python examples/demo_riff.py

Outputs land in examples/out/demo-riff/: stems/, bass.tab.txt, bass.gp5,
drums.musicxml, drums.mid, transcription.json... Drop the generated
"Smells Like Teen Spirit (demo riff).wav" onto the web UI to see the mixer
with a song that transcribes cleanly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drum_extractor.events import DrumHit  # noqa: E402
from drum_extractor.sonify import sonify_drums  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
SONG = "Smells Like Teen Spirit (demo riff)"
SR = 44100
BPM = 117.0
BEAT = 60.0 / BPM
EIGHTH = BEAT / 2
BARS = 8
BAR = 4 * BEAT
TOTAL = BARS * BAR + 1.0

# Two chords per bar, the classic chug pattern on eighths 0,1,3 of each
# half-bar. Bass roots as MIDI: F1=29, Bb1=34, Ab1=32, Db2=37.
CHORDS = [29, 34, 32, 37]
CHUG_EIGHTHS = (0, 1, 3)


def note_times():
    out = []
    for bar in range(BARS):
        for half in range(2):
            root = CHORDS[(bar * 2 + half) % 4]
            base = bar * BAR + half * 2 * BEAT
            out += [(base + e * EIGHTH, root) for e in CHUG_EIGHTHS]
    return out


def synth_tone(f0: float, dur: float, kind: str):
    t = np.arange(int(dur * SR)) / SR
    if kind == "bass":
        wave = np.sign(np.sin(2 * np.pi * f0 * t)) * 0.35 + np.sin(2 * np.pi * f0 * t) * 0.65
        env = np.minimum(1, t / 0.005) * np.exp(-t * 3.5)
        return np.tanh(1.8 * wave * env)
    # "guitar": root + fifth + octave, detuned saws, saturated
    sig = np.zeros_like(t)
    for mult, gain in ((1.0, 1.0), (1.5, 0.8), (2.0, 0.6)):
        for det in (0.997, 1.003):
            ph = 2 * np.pi * f0 * mult * det * t
            sig += gain * (2 * ((ph / (2 * np.pi)) % 1) - 1)
    env = np.minimum(1, t / 0.004) * np.exp(-t * 2.2)
    return np.tanh(1.2 * sig * env / 3.0)


def render_stem(kind: str, octave_up: bool = False):
    y = np.zeros(int(TOTAL * SR), dtype=np.float64)
    for t0, root in note_times():
        f0 = 440.0 * 2 ** ((root + (12 if octave_up else 0) - 69) / 12)
        snd = synth_tone(f0, EIGHTH * 1.8, kind)
        i = int(t0 * SR)
        j = min(i + len(snd), len(y))
        y[i:j] += snd[: j - i]
    return (0.8 * y / (np.max(np.abs(y)) + 1e-9)).astype("float32")


def drum_hits():
    """The published main-groove pattern (per standard drum transcriptions):
    straight-8th hats (crash pulse in the 'chorus' half), backbeat snare on
    2 and 4, kick on 1 — and the signature 16th figure around beat 3: kick on
    3, ghost snare on the 'e', double kick on the '& a'. Bars 4/8 end with the
    16th-snare pickup into the next phrase."""
    hits = []
    for bar in range(BARS):
        b0 = bar * BAR
        chorus = bar >= BARS // 2
        # 8th-note pulse: hats in the first half, crash-ride wash in the second.
        pulse = "crash" if chorus else "hihat_closed"
        for e in range(8):
            vel = 104 if e % 2 == 0 else 84
            hits.append(DrumHit(b0 + e * EIGHTH, pulse, vel if chorus else vel - 8))
        # Kick: 1, then the beat-3 cluster (3, 3&, 3a).
        for beat in (0.0, 2.0, 2.5, 2.75):
            hits.append(DrumHit(b0 + beat * BEAT, "kick", 112))
        # Snare: backbeat on 2 and 4, ghost 16th on the 'e' of 3.
        hits += [DrumHit(b0 + BEAT, "snare", 118),
                 DrumHit(b0 + 2.25 * BEAT, "snare", 42),
                 DrumHit(b0 + 3 * BEAT, "snare", 118)]
        if bar % 4 == 3 and bar != BARS - 1:  # 16th-snare pickup into the next phrase
            hits += [DrumHit(b0 + 3.5 * BEAT, "snare", 96),
                     DrumHit(b0 + 3.75 * BEAT, "snare", 104)]
    return sorted(hits, key=lambda h: h.time)


def main() -> None:
    stems_dir = OUT / SONG / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    sonify_drums(drum_hits(), stems_dir / "drums.wav", sr=SR, tail=1.0)
    sf.write(str(stems_dir / "bass.wav"), render_stem("bass"), SR)
    sf.write(str(stems_dir / "other.wav"), render_stem("guitar", octave_up=True), SR)

    # Pad stems to equal length, like real Demucs output.
    ys = {p: sf.read(str(p))[0] for p in stems_dir.glob("*.wav")}
    n = max(len(y) for y in ys.values())
    for p, y in ys.items():
        if len(y) < n:
            sf.write(str(p), np.pad(y, (0, n - len(y))).astype("float32"), SR)

    mix = sum(sf.read(str(p))[0] for p in stems_dir.glob("*.wav"))
    mix = 0.85 * mix / (np.max(np.abs(mix)) + 1e-9)
    sf.write(str(OUT / f"{SONG}.wav"), mix.astype("float32"), SR)

    from drum_extractor.config import DrumTranscriptionConfig, NotationConfig, PipelineConfig, QuantizeConfig
    from drum_extractor.pipeline import run_pipeline

    config = PipelineConfig(
        output_dir=OUT,
        drums=DrumTranscriptionConfig(backend="onset"),
        quantize=QuantizeConfig(backend="librosa", fixed_tempo=BPM, grid_mode="constant"),
        do_separation=False,
    )
    # Engrave the demo's GROUND-TRUTH chart first (we know the exact hits we
    # synthesized — same trick as the accuracy bank's reference charts). The
    # pipeline's drums.musicxml is the TRANSCRIBER's take, which is honest but
    # rougher; reference.musicxml is what the groove actually is.
    try:
        from drum_extractor.events import Transcription
        from drum_extractor.notation import notate_drums
        from drum_extractor.quantize import quantize as _quantize

        truth = Transcription(drum_hits=drum_hits(), tempo=BPM)
        _quantize(truth, OUT / SONG / "stems" / "drums.wav",
                  QuantizeConfig(backend="librosa", fixed_tempo=BPM, grid_mode="constant"))
        notate_drums(truth, OUT / SONG,
                     NotationConfig(title=f"{SONG} — groove chart", render_pdf=False),
                     QuantizeConfig())
        (OUT / SONG / "drums.musicxml").replace(OUT / SONG / "reference.musicxml")
        print(f"ground-truth chart: {OUT / SONG / 'reference.musicxml'}")
    except Exception as exc:  # notation extra missing — the demo still works
        print(f"(reference chart skipped: {exc})")

    result = run_pipeline(OUT / f"{SONG}.wav", config)
    print(result.summary())
    print(f"\nfull mix to drop on the web UI: {OUT / f'{SONG}.wav'}")


if __name__ == "__main__":
    main()
