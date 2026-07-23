# Examples

### `demo_notation.py` — verify the notation stage without any audio

Builds a synthetic rock groove and engraves it, so you can confirm Stage 4 works
before wiring up the audio models:

```bash
pip install -e ".[notation]"
python examples/demo_notation.py
```

Produces `examples/out/drums.musicxml` (and a PDF if MuseScore's CLI is on
PATH). Open it in any notation editor to see the percussion clef, x-noteheads for
hi-hat/crash, and the hands/feet stem split.

### End-to-end on a real song

```bash
pip install -e ".[all]"      # + `pip install adtof` for the best drums
drum-extractor run /path/to/song.mp3 -o output
```

Then open `output/song/drums.musicxml`, and listen to `output/song/stems/drums.wav`
and `bass.wav`.

> Start with a mid-tempo track to see the pipeline at its best; try a blast-beat
> track to see exactly where transcription hits its limits (see the main README).
