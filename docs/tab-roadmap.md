# Guitar & bass tab improvement roadmap

Findings from a July 2026 survey of open-source projects, papers, and datasets
that could improve our tab pipeline (basic-pitch → chord-aware DP →
ASCII/.gp5). Every repo below was actually fetched and read — licenses were
verified from LICENSE files where the network allowed, and anything assessed
only from secondary sources is flagged.

## Quick wins (days each)

| # | What | Why | Effort |
|---|------|-----|--------|
| 1 | **alphaTab** in-browser tab renderer ([CoderLine/alphaTab](https://github.com/CoderLine/alphaTab), MPL-2.0, v1.8.4 Jul 2026) | Renders our existing `.gp5` as engraved notation+tab SVG; self-hosts offline (~1.5–2.5 MB static assets, zero runtime deps); External Cursor API (v1.6+) can sync a playback cursor to our Web Audio clock — tabs highlight along with the song like the drum sheet already does | 1–2 days viewer, +2–5 days synced cursor |
| 2 | **crepe_notes** bass backend ([xavriley/crepe_notes](https://github.com/xavriley/crepe_notes), GPL-3.0, maintained) | Published +10% F-measure over basic-pitch on source-separated bass stems (FiloBass) — our exact setting; velocity preserved | 0.5–1 day behind the existing `config.backend` switch; benchmark repeated same-pitch pedaling first (its documented weak spot) |
| 3 | **chords-db** voicing snap ([tombatossals/chords-db](https://github.com/tombatossals/chords-db), MIT) | 3,283 hand-curated voicings with resolved MIDI numbers; match detected chord pitch-class sets and inject idiomatic shapes as discounted candidates in `_voicings()` — fixes "voicings don't look like what a guitarist plays" | 1–2 days; add ~a dozen drop-D shapes by hand (DB is E-standard) |
| 4 | **MuScriptor** guitar/bass engine ([muscriptor/muscriptor](https://github.com/muscriptor/muscriptor), MIT code / CC BY-NC weights, v0.2.2 Jul 2026) | Only open engine trained on heavy metal at scale (170k real recordings); onset F1 60.4 vs YourMT3+'s 32.5 on real-music eval — best shot at the distorted-guitar problem. No velocity output | 1–2 days to a verdict; try on full mix and on Demucs stems |

## Soon (about a week each)

- **mt3-infer second-opinion filter** ([openmirlab/mt3-infer](https://github.com/openmirlab/mt3-infer),
  MIT, active): run YourMT3/MR-MT3 next to basic-pitch and keep notes both
  agree on — spectral and token-decoder models make uncorrelated errors, which
  directly attacks harmonic ghost notes on high-gain guitar.
- **Technique markers** (palm mute / bend / slide / vibrato): no adoptable open
  code exists (SoloLa is unlicensed Theano; TART has no code), but the recipe
  is proven — TENT-style contour rules on the torchcrepe output we already
  compute, plus a spectral-centroid/decay palm-mute heuristic on onsets.
  PyGuitarPro already models all these note effects, and alphaTab renders them.
- **Data-driven DP costs**: mine string/fret co-occurrence stats from real
  metal tabs (request [DadaGP](https://dadagp.github.io/), or any local `.gp5`
  pile) à la
  [guitar-transcription-with-inhibition](https://github.com/cwitkowitz/guitar-transcription-with-inhibition)
  (MIT) and blend as a prior into `_voicing_cost()` — replaces hand-tuned
  constants with priors learned from the genre.

## Real-audio accuracy bank for tabs

The drum bank taught us synthetic-only tuning overfits. For tabs the plan is:

1. **GuitarSet** (MIT — the only major set we can bundle a downloader for
   without caveats; fetch via `mirdata`): real audio + per-string note labels.
   Score the full pipeline **and** the DP alone (feed ground-truth notes,
   compare fingerings against real players').
2. Metrics per the literature: note-F via `mir_eval` (50 ms onset / 50 cent
   pitch tolerance), string-aware tab-note-F, and **TDR**
   (tab_precision ÷ pitch_precision) — TDR isolates string-assignment quality
   from transcription quality.
3. Distortion-robustness slice: **Guitar-TECHS** (CC BY 4.0 — cleanest legal
   real distorted-electric source, DI + miked amp + technique labels),
   **EGFxSet** (string+fret labels under real distortion pedals),
   **IDMT-SMT-Audio-Effects** distorted polyphonic chords (CC BY-NC-ND:
   downloader to the official Zenodo URL is fine; never commit audio or
   derived clips — ND forbids distributing modified audio).
4. Bass: **IDMT-SMT-Bass-Single-Track** — the only public electric-bass lines
   with string+fret ground truth; the single-note set stress-tests torchcrepe
   octave refinement per technique (slap vs fingerstyle vs muted).
5. **EGDB** (240 DI-vs-amp excerpts incl. high-gain Mesa renders, no license
   stated, Google-Drive hosted): document a manual fetch as an optional
   extension; don't bundle. Request access to **GOAT** (ISMIR 2025: real DI
   electric + amp renders with full GP tabs incl. techniques) as the long-term
   tuning set.

## Longer term

- **Fretting-Transformer**-style MIDI→tab model (ICMC 2025; no official code —
  [Open-Fret](https://github.com/Sidmaz666/open-fret) is a pre-release
  scaffold) trained on DadaGP-style data: learned genre-idiomatic fingering
  with tuning/capo conditioning, keeping our DP as the deterministic fallback.
  Only if the cheap DP upgrades plateau (1–3 weeks + GPU).
- **EGDB-PG recipe** for high-gain robustness at the source: fine-tune a note
  front end on clean DI datasets re-rendered through amp sims (the
  Tone-informed Transformer paper, arXiv 2504.07406, validates this is *the*
  fix for high-gain transcription). Dataset is available on request only.
- Print-quality engraved tab PDFs via music21 MusicXML → headless MuseScore
  (music21 exports `<string>`/`<fret>` but `<staff-tuning>`/`<capo>` are TODO
  upstream — needs a small ElementTree shim for drop tunings).

## Verified skips (so we don't re-litigate)

- **TabCNN** — no license file, Python 2.7, no weights, superseded by FretNet.
- **FretNet / amt-tools as engines** — MIT but no released weights; training
  is weeks on acoustic-only GuitarSet (domain mismatch). The inhibition-matrix
  *idea* is adopted above instead.
- **TuxGuitar's MIDI importer** — read the source: greedy per-beat
  closest-string allocation, strictly less capable than our DP. (TuxGuitar the
  *app* is great — the maintained fork
  [helge17/tuxguitar](https://github.com/helge17/tuxguitar) v2.1.0 is the free
  desktop editor to recommend for our `.gp5` downloads.)
- **Omnizart** — revived May 2026 but the music model is 2021-era, and it
  drags TensorFlow + madmom into a torch stack.
- **VexFlow/VexTab** — rendering primitives only (we'd rebuild what alphaTab
  ships) / non-commercial license.
- **Verovio for tabs** — its tablature support targets historical lute
  notation; guitar staff-tab is experimental as of v6.x. (It stays as our drum
  sheet renderer.)
- **SoloLa** — unlicensed, dead Theano stack; its contour-classification
  recipe lives on in the technique-markers item above.
- **FiloBass** — restricted research terms, upright jazz bass.
- **SynthTab** — synthetic audio for *evaluation* is the exact overfitting
  trap the drum bank already caught; training augmentation only.
