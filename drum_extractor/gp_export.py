"""Guitar Pro (.gp5) export for the bass and guitar transcriptions.

Tabs as ASCII are readable, but ``.gp5`` is the lingua franca for
guitarists/bassists — Guitar Pro, TuxGuitar and MuseScore all open it, with
playback. Notes are quantized onto a 16th grid at the detected tempo (drums
philosophy: onsets matter, durations are grid slots).

Optional dependency: PyGuitarPro (the ``gp`` extra). Callers should treat a
missing library as "skip the export", which the pipeline does.
"""

from __future__ import annotations

from pathlib import Path

from .errors import MissingDependencyError
from .events import BassNote
from .logging_utils import get_logger

log = get_logger(__name__)

SLOTS_PER_MEASURE = 16  # 4/4 sixteenths


def write_gp5(
    notes: list[BassNote],
    tuning: tuple[int, ...],
    path: str | Path,
    tempo: float | None = None,
    title: str = "",
    track_name: str = "Track",
    instrument: int = 33,
) -> Path:
    """Write assigned notes (string/fret set) to a Guitar Pro 5 file."""
    try:
        import guitarpro as gp
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Guitar Pro export", "PyGuitarPro", extra="gp") from exc

    bpm = float(tempo) if tempo and tempo > 0 else 120.0
    slot_s = (60.0 / bpm) * 4.0 / SLOTS_PER_MEASURE  # sixteenth length in seconds

    song = gp.models.Song()
    song.title = title
    song.tempo = int(round(bpm))
    track = song.tracks[0]
    track.name = track_name
    track.channel.instrument = instrument
    n = len(tuning)
    # Guitar Pro numbers strings from 1 = highest; ours index from 0 = lowest.
    track.strings = [gp.models.GuitarString(number=i + 1, value=tuning[n - 1 - i]) for i in range(n)]

    # Quantize onsets to slots; simultaneous notes share a beat (chord).
    events: dict[int, list[BassNote]] = {}
    placed = 0
    for note in notes:
        if note.string is None or note.fret is None:
            continue  # unplaceable notes live in the MIDI/ASCII outputs
        events.setdefault(int(round(note.start / slot_s)), []).append(note)
        placed += 1
    if not placed:
        raise ValueError("No fretted notes to export.")

    n_measures = max(events) // SLOTS_PER_MEASURE + 1
    while len(song.measureHeaders) < n_measures:
        header = gp.models.MeasureHeader()
        header.number = len(song.measureHeaders) + 1
        song.measureHeaders.append(header)
        for tr in song.tracks:
            tr.measures.append(gp.models.Measure(tr, header))

    for mi in range(n_measures):
        voice = track.measures[mi].voices[0]
        for s in range(SLOTS_PER_MEASURE):
            beat = gp.models.Beat(voice)
            beat.duration = gp.models.Duration(value=SLOTS_PER_MEASURE)
            here = events.get(mi * SLOTS_PER_MEASURE + s, [])
            if here:
                beat.status = gp.models.BeatStatus.normal
                used: set[int] = set()
                for bn in here:
                    gp_string = n - bn.string  # type: ignore[operator]
                    if gp_string in used:
                        continue  # two quantized onto one string: keep the first
                    used.add(gp_string)
                    gn = gp.models.Note(beat)
                    gn.type = gp.models.NoteType.normal
                    gn.string = gp_string
                    gn.value = int(bn.fret)  # type: ignore[arg-type]
                    gn.velocity = int(max(15, min(127, bn.velocity)))
                    beat.notes.append(gn)
            else:
                beat.status = gp.models.BeatStatus.rest
            voice.beats.append(beat)

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
