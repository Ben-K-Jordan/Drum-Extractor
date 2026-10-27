"""Guitar Pro (.gp5) export for the bass and guitar transcriptions.

Tabs as ASCII are readable, but ``.gp5`` is the lingua franca for
guitarists/bassists — Guitar Pro, TuxGuitar and MuseScore all open it, with
playback. Onsets are quantized onto a 16th grid at the detected tempo; each
note then lasts until the next onset (or its own end), never past the next beat
line, and the gaps become rests that don't cross one either, so the staff reads
as real rhythm (an eighth-note chug, not a 16th and a rest).

Optional dependency: PyGuitarPro (the ``gp`` extra). Callers should treat a
missing library as "skip the export", which the pipeline does.
"""

from __future__ import annotations

from pathlib import Path

from .errors import MissingDependencyError
from .events import BassNote
from .logging_utils import get_logger

log = get_logger(__name__)

SIXTEENTHS_PER_BEAT = 4  # slot resolution: sixteenths
# Note values a run of 1-4 sixteenth slots inside one beat can take.
_SLOT_VALUES = {1: (16, False), 2: (8, False), 3: (8, True), 4: (4, False)}


def _rest_runs(start: int, stop: int):
    """Split the empty slots [start, stop) into rests that stay inside a beat,
    largest first where they line up (quarter on the beat, eighth on an 8th)."""
    pos = start
    while pos < stop:
        if pos % SIXTEENTHS_PER_BEAT == 0 and stop - pos >= 4:
            run = 4
        elif pos % 2 == 0 and stop - pos >= 2 and (pos % SIXTEENTHS_PER_BEAT) + 2 <= SIXTEENTHS_PER_BEAT:
            run = 2
        else:
            run = 1
        yield pos, run
        pos += run


def write_gp5(
    notes: list[BassNote],
    tuning: tuple[int, ...],
    path: str | Path,
    tempo: float | None = None,
    title: str = "",
    track_name: str = "Track",
    instrument: int = 33,
    artist: str = "",
    beats_per_bar: int = 4,
) -> Path:
    """Write assigned notes (string/fret set) to a Guitar Pro 5 file.

    ``beats_per_bar`` sets the measure length and time-signature numerator
    (the denominator stays 4 — the slot grid is sixteenths per quarter).
    """
    try:
        import guitarpro as gp
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Guitar Pro export", "PyGuitarPro", extra="gp") from exc

    bpm = float(tempo) if tempo and tempo > 0 else 120.0
    slot_s = 60.0 / bpm / SIXTEENTHS_PER_BEAT  # sixteenth length in seconds
    # Clamp ONCE and use everywhere: a raw beats_per_bar=0 would silently
    # write an invalid 0/4 time signature while the grid used 1 beat.
    beats_per_bar = max(1, int(beats_per_bar))
    slots_per_measure = beats_per_bar * SIXTEENTHS_PER_BEAT

    song = gp.models.Song()
    song.title = title
    song.artist = artist
    song.tempo = int(round(bpm))
    track = song.tracks[0]
    track.name = track_name
    track.channel.instrument = instrument
    n = len(tuning)
    # Guitar Pro numbers strings from 1 = highest; ours index from 0 = lowest.
    track.strings = [gp.models.GuitarString(number=i + 1, value=tuning[n - 1 - i]) for i in range(n)]

    # Quantize onsets to slots; simultaneous notes share a beat (chord). Two
    # notes on the SAME string in the same slot cannot both engrave there —
    # bump the later one to the next slot where its string is free (a real
    # fast-repeated note survives; silently dropping it would desync the gp5
    # from the MIDI/ASCII outputs). Negative onsets clamp to slot 0.
    events: dict[int, list[BassNote]] = {}
    placed = skipped = 0
    for note in sorted(notes, key=lambda x: (x.start, x.pitch)):
        if note.string is None or note.fret is None:
            continue  # unplaceable notes live in the MIDI/ASCII outputs
        slot = max(0, int(round(note.start / slot_s)))
        limit = slot + slots_per_measure  # bump at most one bar before giving up
        while slot < limit and any(b.string == note.string for b in events.get(slot, [])):
            slot += 1
        if slot >= limit:
            skipped += 1
            continue
        events.setdefault(slot, []).append(note)
        placed += 1
    if not placed:
        raise ValueError("No fretted notes to export.")
    if skipped:
        log.warning("Guitar Pro export: %d note(s) had no free slot/string and were omitted", skipped)

    n_measures = max(events) // slots_per_measure + 1
    for header in song.measureHeaders:
        header.timeSignature.numerator = beats_per_bar
    while len(song.measureHeaders) < n_measures:
        header = gp.models.MeasureHeader()
        header.number = len(song.measureHeaders) + 1
        header.timeSignature.numerator = beats_per_bar
        song.measureHeaders.append(header)
        for tr in song.tracks:
            tr.measures.append(gp.models.Measure(tr, header))

    def add_beat(voice, slots: int, here=()) -> None:
        value, dotted = _SLOT_VALUES[slots]
        beat = gp.models.Beat(voice)
        beat.duration = gp.models.Duration(value=value, isDotted=dotted)
        beat.status = gp.models.BeatStatus.normal if here else gp.models.BeatStatus.rest
        for bn in here:
            gn = gp.models.Note(beat)
            gn.type = gp.models.NoteType.normal
            gn.string = n - bn.string  # type: ignore[operator]
            gn.value = int(bn.fret)  # type: ignore[arg-type]
            gn.velocity = int(max(15, min(127, bn.velocity)))
            beat.notes.append(gn)
        voice.beats.append(beat)

    for mi in range(n_measures):
        voice = track.measures[mi].voices[0]
        m0, m1 = mi * slots_per_measure, (mi + 1) * slots_per_measure
        onsets = sorted(s for s in events if m0 <= s < m1)
        pos = m0
        for k, slot in enumerate(onsets):
            for _, run in _rest_runs(pos - m0, slot - m0):
                add_beat(voice, run)
            nxt = onsets[k + 1] if k + 1 < len(onsets) else m1
            beat_end = (slot // SIXTEENTHS_PER_BEAT + 1) * SIXTEENTHS_PER_BEAT
            own = max(max(1, int(round((bn.end - bn.start) / slot_s))) for bn in events[slot])
            length = max(1, min(nxt - slot, beat_end - slot, own))
            add_beat(voice, length, events[slot])
            pos = slot + length
        for _, run in _rest_runs(pos - m0, m1 - m0):
            add_beat(voice, run)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gp.write(song, str(path))
    log.info("Wrote Guitar Pro tab: %s (%d notes, %d measures)", path, placed, n_measures)
    return path


def gp_available() -> bool:
    try:
        import guitarpro  # noqa: F401

        return True
    except Exception:
        return False
