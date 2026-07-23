# Development notes

How this project is built and verified. The short version: every substantial
change goes through adversarial review, and transcription accuracy is a
measured, CI-gated number — not a hope.

## Quality process

Each development round followed the same loop:

1. **Implement** against the real upstream libraries (never just their docs).
2. **Adversarial audit** — parallel reviewers statically read every module and
   dynamically attack it with edge cases in a real-dependency environment;
   every finding is then independently verified to kill false positives.
3. **Fix + regression-test** everything confirmed.

What that process caught, by round:

### Round 1 — running the code instead of trusting it
- A NumPy-2 crash in the librosa tempo path (`float()` on an array).
- Duplicate hits when two onsets snapped to the same grid slot.

### Round 2 — full adversarial audit (16 confirmed defects)
- Songs starting on a pickup produced a negative beat that **crashed the whole
  notation stage** on the default backend.
- Barline-crossing notes produced overfull/malformed measures.
- The CREPE octave-refinement used `fmin=30`, below torchcrepe's lowest bin,
  collapsing the pitch track and corrupting correct notes.
- The onset fallback fabricated phantom kicks/snares on ~half of all onsets
  (median self-referencing threshold).
- Plus 12 more (last-bar tempo assumption, non-quarter meters, tempo-less MIDI
  notation, IR persistence ordering, ...).

### Round 3 — comparison against the upstream open-source projects
Each module was diffed against the real source of Demucs, UVR,
python-audio-separator, ADTOF / ADTOF-pytorch, basic-pitch, torchcrepe,
music21, madmom and note-seq. Most API usage was validated as correct; the
critical catch: **the default ADTOF command matched no real tool** (`python -m
adtof` doesn't exist), so the flagship backend silently fell back to the crude
onset detector. Also fixed: unaligned ensemble averaging (comb-filtered the
transients it was meant to clean), a cosmetic-only `fixed_tempo`, and a
periodicity-less CREPE median.

### Round 4 — the groove bank and the overfitting lesson
The ground-truth bank (see below) exposed that the onset fallback's thresholds
could never fire on a snare. A first re-tune then **overfit the synthesizer's
own spectra** — caught by the next verification round, which showed phantom
instruments firing on realistic drum sounds while the regression test passed
vacuously. The shipped fix: realistic synthesis voices, a dominance-based
classifier, and thresholds grid-searched **under hard constraints from held-out
realistic proxies** with a safety margin.

## The ground-truth groove bank

`drum-extractor bank-build` generates grooves programmatically (rock 8ths,
punk d-beat, 200 BPM double-kick, 220 BPM blast, gallops, fills, ghost notes),
renders them with the project's own drum synthesizer, and stores the exact hits.
`drum-extractor bank-eval` transcribes that audio and scores it (±50 ms
per-family F-measure) — so accuracy can be measured, tuned, and
regression-tested without any manual annotation. Every item is also an
(audio, ground-truth MIDI) pair, i.e. ready-made fine-tuning data.

Current onset-fallback scores (the built-in dependency-light backend — ADTOF
is the serious transcriber):

| | cymbals | hihat | kick | snare | toms | overall |
|---|---|---|---|---|---|---|
| intuition-set thresholds | 0.00 | 0.82 | 0.12 | 0.00 | 0.00 | **0.46** |
| shipped (robust-tuned) | 0.00 | 0.86 | 0.71 | 0.71 | 0.00 | **0.73** |

The bank also quantified the double-kick booster: kick recall 0.67 → 0.98 on
200 BPM sixteenth double-kick, at the cost of aggregate F elsewhere — hence
it's a per-song flag, not a default.

## CI

`.github/workflows/ci.yml` runs two jobs on every push:

- **Test suite** — the full pytest suite with the drums/notation/web extras
  (heavy ML deps deliberately absent; their tests skip by design).
- **Accuracy gate** — rebuilds the groove bank, scores the onset backend, and
  fails the build if aggregate F drops below 0.65.

## Environment sandbox note

This project was developed in a sandbox whose egress policy blocks the model
weight hosts (dl.fbaipublicfiles.com, huggingface.co), so Demucs' neural
inference itself is exercised on user machines rather than in CI; its API
usage, tensor→file plumbing, and CLI fallback are covered by tests, and the
whole UI/transcription/notation surface runs against real synthesized audio.
