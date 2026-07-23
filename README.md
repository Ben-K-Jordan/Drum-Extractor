# Drum-Extractor

Give it a metal, punk, or rock song and it returns the **isolated drums and bass** —
then transcribes the drums into **MIDI and drum sheet music** for practice. Built
around mature open-source models, with every stage designed for the messy reality
of loud, distorted, fast music.

> **Is this realistic?** Yes — with an honest split.
> - **Isolating drums + bass is a solved problem.** Demucs v4 gives clean stems from a full mix. This works today.
> - **Drum sheet music is workable-to-experimental.** A normal groove transcribes to a genuinely useful chart. Fast metal (blast beats, 200+ BPM double-bass) comes out as a *rough scaffold you correct by ear* — that's a limit of every current model, not this tool. Set expectations accordingly.
> - **Bass transcription** is the easiest of the three (bass is mostly monophonic).

---

## The pipeline

```
  full song (wav/mp3/flac)
          │
          ▼
  ┌──────────────────────────┐
  │ Stage 1  SEPARATION      │   Demucs v4 (htdemucs_ft)        [MIT]
  │ full mix → drums + bass  │   → drums.wav, bass.wav
  └──────────────────────────┘
       │ drums.wav                 │ bass.wav
       ▼                           ▼
  ┌──────────────────┐    ┌──────────────────────┐
  │ Stage 2a  DRUMS  │    │ Stage 2b  BASS       │
  │ ADTOF → drum MIDI│    │ basic-pitch → MIDI   │   [Apache-2.0]
  │ (onset fallback) │    │ (+ torchcrepe octave │
  │        [NC*]     │    │  correction, + tab)  │
  └──────────────────┘    └──────────────────────┘
       │ drum hits
       ▼
  ┌──────────────────────────┐
  │ Stage 3  QUANTIZE        │   madmom beat/tempo (librosa    [BSD]
  │ snap onsets to bar grid  │   fallback)
  └──────────────────────────┘
       │
       ▼
  ┌──────────────────────────┐
  │ Stage 4  NOTATION        │   music21 → MusicXML → PDF      [BSD/GPL]
  │ → drum sheet music       │   (MuseScore CLI renders PDF)
  └──────────────────────────┘

  * ADTOF weights are non-commercial (fine for personal learning).
    A commercial-safe swap is OaF-Drums (Apache-2.0).
```

Each stage runs independently and **degrades gracefully**: if an optional
dependency isn't installed, that stage is skipped with a clear message telling
you exactly what to `pip install`, while the rest of the pipeline still runs.
The package imports with zero heavy dependencies.

---

## Install

**Quick start — just the stem splitter (the reliable MVP):**

```bash
pip install -e .                 # core: numpy + pretty_midi
pip install -e ".[separation]"   # adds Demucs (pulls torch/torchaudio)
```

**Full pipeline:**

```bash
pip install -e ".[all]"          # separation + drums(onset) + bass + notation
```

Three tools install separately because they're heavier or app-based:

| Tool | Why separate | Install |
|------|--------------|---------|
| **ADTOF** (best drum transcriber) | own heavy deps | `pip install adtof` (see notes below) |
| **madmom** (beat tracking) | can be tricky to build; auto-falls back to librosa | `pip install ".[quantize]"` |
| **MuseScore 4** (PDF rendering) | desktop app, not a pip package | install from [musescore.org](https://musescore.org), ensure `mscore`/`musescore4` is on PATH |

> A GPU makes Stage 1 much faster but isn't required — Demucs runs on CPU at
> roughly 1–2× the track length per song.

---

## Usage

### Command line

```bash
# Full pipeline
drum-extractor run song.mp3

# Just isolate the stems (Stage 1)
drum-extractor separate song.mp3 --stems drums,bass

# Individual stages
drum-extractor transcribe-drums output/song/stems/drums.wav
drum-extractor transcribe-bass  output/song/stems/bass.wav
drum-extractor notate           output/song/transcription.json

# Fast-metal tips: finer grid for double-kick, and the onset fallback if you
# haven't installed ADTOF yet
drum-extractor run song.mp3 --grid 32 --drum-backend onset
```

Outputs land in `output/<song>/`:

```
output/song/
├── stems/drums.wav        # Stage 1
├── stems/bass.wav
├── drums.mid              # Stage 2a
├── bass.mid               # Stage 2b
├── bass.tab.txt
├── transcription.json     # full intermediate representation
└── drums.musicxml         # Stage 4 (+ drums.pdf if MuseScore is installed)
```

### Python API

```python
from drum_extractor import run_pipeline, PipelineConfig, SeparationConfig, QuantizeConfig

config = PipelineConfig(
    separation=SeparationConfig(model="htdemucs_ft", device="auto"),
    quantize=QuantizeConfig(grid=32),   # finer grid for fast double-kick
)
result = run_pipeline("song.mp3", config)
print(result.summary())
print(result.stems.drums, result.drum_midi, result.musicxml)
```

Or drive one stage at a time:

```python
from drum_extractor.separation import separate
from drum_extractor.drums import transcribe_drums
from drum_extractor.notation import notate_drums

stems = separate("song.mp3", "out/stems")
hits = transcribe_drums(stems.drums)
# ... quantize, then:
# notate_drums(transcription, "out")
```

---

## What to expect on metal / punk / rock

**Where it shines**
- **Drum stem** — the most robust stem for any separator, even on dense distorted mixes. Clean enough to practise against and to feed the transcriber.
- **Bass stem** — the fundamental isolates cleanly under loud guitars; you'll clearly hear the line.
- **Mid-tempo grooves** — kick/snare/hi-hat transcribe to a readable chart with light cleanup.
- **Bass tab** — 4 strings, mostly single notes → clean, playable tab.

**Where it struggles (be honest with yourself)**
- **Fast double-bass / blast beats (200–280 BPM)** — hits ~50 ms apart merge; models *undercount kicks*. This is the single biggest limitation and it hits exactly the passages you most want charted.
- **Simultaneous hits** (kick+snare+crash) and **cymbal/tom identity** are the weak spots of every model.
- **Down-tuned distorted bass** loses its grind in separation and collides with the kick in the sub-bass; extracted pitch is *approximate* (5-string low B is below reliable pitch-tracking range).
- **Articulations** — open/closed hi-hat, ghost notes, flams, chokes mostly don't survive to notation.

**One line to internalise:** normal groove → trustworthy draft; fast metal →
skeleton you verify by ear. No 2026 tool escapes this.

**Levers that help:** always transcribe from the isolated drum stem (never the
mix); use `--grid 32` for double-kick; feed the cleanest/DI bass you have.

---

## Implementation status

| Stage | Status | Notes |
|-------|--------|-------|
| 1. Separation | **Working** | Demucs Python API + CLI fallback, device auto-detect |
| 2a. Drum transcription | **Working** | ADTOF backend + librosa onset fallback |
| 2b. Bass transcription | **Working** | basic-pitch + optional torchcrepe octave fix + tab mapper |
| 3. Quantization | **Working** | madmom with librosa fallback |
| 4. Notation | **Working** | music21 → MusicXML (validated); PDF via MuseScore CLI |

Everything is wired and unit-tested (`pytest`). The parts that need real audio
and the heavier models are best validated on your own machine with the full
extras installed.

### Known rough edges / good first customizations
- **Drum durations & quantization** — notes last until the next onset in their voice; the grid choice (`--grid`) dominates readability. Tune per song.
- **Kit mapping** — edit `PLACEMENT` / `CANONICAL_TO_GM` in `drum_extractor/gm_drum_map.py` to match your kit or notation preferences.
- **ADTOF command** — the exact CLI varies by install; override `DrumTranscriptionConfig.adtof_command` if needed (`{input}`/`{output}` are substituted).
- **Voice numbering** — music21 exports voices as 0/1; MuseScore maps them fine, but you can post-process the MusicXML if you need strict 1/2.

---

## Roadmap

- [x] **Phase 0** — Stem splitter MVP (Demucs wrapper + CLI)
- [x] **Phase 1** — Bass transcription + tab (basic-pitch)
- [x] **Phase 2** — Drum transcription MVP (ADTOF + onset fallback)
- [x] **Phase 3** — Notation / PDF (music21 + MuseScore)
- [ ] **Phase 4** — Metal quality upgrades: RoFormer/SCNet drum-stem ensemble, LarsNet kick isolation for double-bass, a by-ear correction UI
- [ ] **Phase 5** — Optional web UI (fork of spleeter-web) or a play-along practice view

---

## Licenses & credits

Code here is MIT. It orchestrates other people's models — respect their licenses:

- **Demucs** — MIT (facebookresearch/demucs)
- **ADTOF** — non-commercial (CC BY-NC-SA / research); fine for personal learning. Commercial-safe alternative: **OaF-Drums** (Apache-2.0).
- **basic-pitch** — Apache-2.0 (Spotify) · **torchcrepe** — MIT
- **madmom** — BSD (some modules academic-only) · **librosa** — ISC
- **music21** — BSD · **MuseScore** — GPL-3 · **LilyPond/Verovio** — GPL/LGPL

If you ever make this commercial, audit model *weights* specifically (some are
trained on research-only datasets even when the code is permissive).
