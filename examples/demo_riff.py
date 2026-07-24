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
    hits = []
    for bar in range(BARS):
        b0 = bar * BAR
        if bar % 4 == 0:
            hits.append(DrumHit(b0, "crash", 118))
        for e in range(8):
            hits.append(DrumHit(b0 + e * EIGHTH, "hihat_closed", 96 if e % 2 == 0 else 72))
        hits += [DrumHit(b0, "kick", 112), DrumHit(b0 + 1.75 * BEAT, "kick", 100),
                 DrumHit(b0 + 2.5 * BEAT, "kick", 108),
                 DrumHit(b0 + BEAT, "snare", 114), DrumHit(b0 + 3 * BEAT, "snare", 114)]
        if bar % 4 == 3:  # snare fill into the next phrase
            hits += [DrumHit(b0 + 3.5 * BEAT + k * EIGHTH / 2, "snare", 70 + 12 * k) for k in range(4)]
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

    from drum_extractor.config import DrumTranscriptionConfig, PipelineConfig, QuantizeConfig
    from drum_extractor.pipeline import run_pipeline

    config = PipelineConfig(
        output_dir=OUT,
        drums=DrumTranscriptionConfig(backend="onset"),
        quantize=QuantizeConfig(backend="librosa", fixed_tempo=BPM, grid_mode="constant"),
        do_separation=False,
    )
    result = run_pipeline(OUT / f"{SONG}.wav", config)
    print(result.summary())
    print(f"\nfull mix to drop on the web UI: {OUT / f'{SONG}.wav'}")


if __name__ == "__main__":
    main()
