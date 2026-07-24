"""Stage 4 — drum notation (transcription -> MusicXML -> engraved PDF).

Builds a real drum staff with music21: percussion clef, the conventional
notehead per instrument, and a two-voice split (hands stems-up, feet stems-down).
Exports MusicXML (the reliable interchange format) and optionally renders a PDF
by shelling out to the MuseScore CLI.

The engraving itself is solved; the weak link is how clean the *input* rhythm is.
Expect a readable draft for normal grooves and a chart that needs manual tidying
for busy metal.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .config import NotationConfig, QuantizeConfig
from .errors import ExternalToolError, MissingDependencyError
from .events import Transcription
from .gm_drum_map import PLACEMENT, SNARE
from .logging_utils import get_logger

log = get_logger(__name__)

GHOST_VELOCITY = 45  # at or below: parenthesized ghost note
ACCENT_VELOCITY = 110  # at or above: accent mark


def _m21():
    try:
        import music21  # type: ignore
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Notation", "music21", extra="notation") from exc
    return music21


def _offset_ql(hit, beats_per_bar: int, beat_unit: int, tempo: float | None) -> float:
    """Quarter-length offset of a hit from the start of the piece.

    ``hit.beat`` is in beat units (0..beats_per_bar); one beat is ``4/beat_unit``
    quarter-lengths, so compound/non-quarter meters (6/8, cut time) are placed
    correctly rather than assuming beat == quarter.
    """
    if hit.bar is not None and hit.beat is not None:
        ql_per_beat = 4.0 / beat_unit
        return ((hit.bar - 1) * beats_per_bar + hit.beat) * ql_per_beat
    # No bar/beat annotation (quantization skipped): derive from wall-clock time.
    # tempo/BPM is quarter-note based, so seconds*bpm/60 is already quarter-lengths.
    bpm = tempo or 120.0
    return hit.time * bpm / 60.0


def build_score(transcription: Transcription, config: NotationConfig | None = None, quantize: QuantizeConfig | None = None):
    """Build a music21 ``Score`` for the drum part."""
    config = config or NotationConfig()
    quantize = quantize or QuantizeConfig()
    m21 = _m21()
    from music21 import stream, clef, meter, instrument, note, tempo as m21tempo, metadata, duration  # type: ignore

    try:
        from music21 import percussion  # type: ignore
        has_perc_chord = hasattr(percussion, "PercussionChord")
    except Exception:  # pragma: no cover
        percussion = None
        has_perc_chord = False

    beats_per_bar, beat_unit = transcription.time_signature or (quantize.beats_per_bar, quantize.beat_unit)
    grid_ql = 4.0 / quantize.grid

    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = config.title
    score.metadata.composer = "Drum-Extractor"

    part = stream.Part()
    part.partName = "Drums"
    # The abbreviation prints on every system AFTER the first: music21's
    # generic Percussion carries "Perc", which made continuation systems look
    # like a separate percussion part instead of the same drum staff.
    part.partAbbreviation = "Dr."
    kit = instrument.Percussion()
    kit.instrumentName = "Drums"
    kit.instrumentAbbreviation = "Dr."
    part.insert(0, kit)
    part.insert(0, clef.PercussionClef())
    part.insert(0, meter.TimeSignature(f"{beats_per_bar}/{beat_unit}"))
    if transcription.tempo:
        # transcription.tempo is the BEAT-UNIT pulse rate (quantize tracks
        # beats_per_bar pulses/bar), so the mark must reference that unit:
        # 6/8 tracked at eighth=178 would otherwise print "quarter = 178".
        part.insert(0, m21tempo.MetronomeMark(
            number=round(transcription.tempo), referent=duration.Duration(4.0 / beat_unit)
        ))

    # Group hits by (voice, snapped quarter-length offset).
    groups: dict[tuple[int, float], list] = defaultdict(list)
    for hit in transcription.drum_hits:
        place = PLACEMENT.get(hit.instrument, PLACEMENT[SNARE])
        off = _offset_ql(hit, beats_per_bar, beat_unit, transcription.tempo)
        off = round(off / grid_ql) * grid_ql
        groups[(place.voice, off)].append(hit)

    # Unquantized inputs (`notate file.mid`, backend='none') bypass
    # quantize._dedupe_hits, so a flam/doubled hit would engrave as two
    # identical noteheads on the same staff position. Keep the loudest.
    for key, hit_list in groups.items():
        best: dict[str, object] = {}
        for h in hit_list:
            prev = best.get(h.instrument)
            if prev is None or h.velocity > prev.velocity:  # type: ignore[union-attr]
                best[h.instrument] = h
        groups[key] = list(best.values())

    # Integer voice ids export as MusicXML <voice>1</voice> / <voice>2</voice>,
    # which is what MuseScore expects for the hands/feet stem split.
    voices: dict[int, "stream.Voice"] = {1: stream.Voice(id=1), 2: stream.Voice(id=2)}

    # A drum note lasts until the next onset in its own voice (standard drum
    # notation), so straight 8ths read as 8ths rather than 16th-plus-rest — but
    # capped at ONE BEAT: drums are attacks, and a kick two beats before the
    # next one should engrave as a quarter note + rests, not a half note. Also
    # capped at the space remaining in the bar: music21's makeMeasures does NOT
    # split an over-long note into tied notes; it packs the overflow into the
    # same measure, producing an overfull/malformed bar. The last onset in a
    # voice defaults to a one-beat length rather than a lone 1/grid sliver.
    bar_ql = beats_per_bar * (4.0 / beat_unit)
    ql_per_beat = 4.0 / beat_unit
    offsets_by_voice: dict[int, list[float]] = defaultdict(list)
    for (voice_id, off) in groups:
        offsets_by_voice[voice_id].append(off)
    dur_map: dict[tuple[int, float], float] = {}
    for vid, offs in offsets_by_voice.items():
        offs.sort()
        for i, off in enumerate(offs):
            gap = (offs[i + 1] - off) if i + 1 < len(offs) else ql_per_beat
            dur = min(ql_per_beat, max(grid_ql, round(gap / grid_ql) * grid_ql))
            pos_in_bar = off - (int(off // bar_ql) * bar_ql)
            remaining = bar_ql - pos_in_bar
            if remaining >= grid_ql - 1e-9:
                dur = min(dur, remaining)
            dur_map[(vid, off)] = dur

    for (voice_id, off), hits in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        elements = []
        stem_dir = "up" if voice_id == 1 else "down"
        note_dur = duration.Duration(dur_map[(voice_id, off)])
        from music21 import articulations  # type: ignore

        for hit in hits:
            place = PLACEMENT.get(hit.instrument, PLACEMENT[SNARE])
            u = note.Unpitched()
            u.displayStep = place.step
            u.displayOctave = place.octave
            u.notehead = place.notehead
            u.stemDirection = place.stem
            u.volume.velocity = hit.velocity
            # Dynamics are half the groove: quiet hits engrave as ghost notes
            # (parenthesized), loud ones get an accent mark.
            if hit.velocity <= GHOST_VELOCITY:
                u.noteheadParenthesis = True
            elif hit.velocity >= ACCENT_VELOCITY:
                u.articulations.append(articulations.Accent())
            elements.append((u, place))

        if len(elements) == 1:
            el = elements[0][0]
            el.duration = duration.Duration(note_dur.quarterLength)
            voices[voice_id].insert(off, el)
        elif has_perc_chord:
            chord = percussion.PercussionChord([e for e, _ in elements])
            chord.duration = duration.Duration(note_dur.quarterLength)
            chord.stemDirection = stem_dir
            # music21 exports only the CHORD's articulations — accents left on
            # member notes vanish (e.g. an accented snare under a hi-hat).
            chord.articulations = [a for el, _ in elements for a in el.articulations]
            voices[voice_id].insert(off, chord)
        else:  # fallback: insert individually (may collide visually)
            for el, _ in elements:
                el.duration = duration.Duration(note_dur.quarterLength)
                voices[voice_id].insert(off, el)

    for vid in (1, 2):
        if voices[vid].notes:
            part.insert(0, voices[vid])

    score.insert(0, part)
    try:
        score.makeMeasures(inPlace=True)
        score.makeRests(fillGaps=True, inPlace=True)
        _renumber_voices(score)
    except Exception as exc:  # pragma: no cover - music21 can be picky on messy input
        log.warning("makeMeasures/makeRests warning: %s", exc)
    return score


def _renumber_voices(score) -> None:
    """Give measure voices the conventional 1-based MusicXML ids.

    music21's makeMeasures re-creates Voice objects with 0-based ids, exporting
    ``<voice>0</voice>``. MuseScore copes, but the drum-notation convention is
    voice 1 = hands (stems up), voice 2 = feet (stems down) — recover that from
    the stem directions we set on every note.
    """
    from music21 import stream  # type: ignore

    for measure in score.recurse().getElementsByClass("Measure"):
        for v in measure.getElementsByClass(stream.Voice):
            notes = list(v.notes)
            all_down = bool(notes) and all(getattr(n, "stemDirection", None) == "down" for n in notes)
            v.id = "2" if all_down else "1"


def transcription_to_musicxml(transcription: Transcription, out_path: str | Path, config: NotationConfig | None = None, quantize: QuantizeConfig | None = None) -> Path:
    """Build the score and write it to a MusicXML file."""
    score = build_score(transcription, config, quantize)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(out_path))
    _unhide_rests(out_path)
    log.info("Wrote MusicXML: %s", out_path)
    return out_path


def _unhide_rests(musicxml_path: Path) -> None:
    """Make every padding rest visible in the written MusicXML.

    music21's EXPORT-side makeNotation marks trailing padding rests
    ``print-object="no"``, so a bar could show a rest on beat 2 but blank
    space on beat 4. Standard drum engraving shows every counted beat. This
    must post-process the file — the hidden rests do not exist yet on the
    Score object that build_score returns.
    """
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(musicxml_path)
        changed = False
        for note in tree.getroot().iter("note"):
            if note.get("print-object") == "no" and note.find("rest") is not None:
                del note.attrib["print-object"]
                changed = True
        if changed:
            tree.write(musicxml_path, encoding="utf-8", xml_declaration=True)
    except ET.ParseError as exc:  # never fail notation over a cosmetic pass
        log.warning("Could not post-process rests in %s: %s", musicxml_path.name, exc)


# Common MuseScore executable names across platforms/packagings. doctor and
# render_pdf MUST resolve from the same list: probing more names than the
# renderer uses meant doctor could report "ok" for a binary PDF export never
# tried.
MUSESCORE_CANDIDATES = ("mscore", "musescore", "musescore4", "mscore4portable", "MuseScore4.exe")


def find_musescore(preferred: str | None = None) -> str | None:
    """Resolve the MuseScore CLI: the configured name first, then common names."""
    import shutil

    for cand in ([preferred] if preferred else []) + list(MUSESCORE_CANDIDATES):
        exe = shutil.which(cand)
        if exe:
            return exe
    return None


def render_pdf(musicxml_path: str | Path, pdf_path: str | Path, config: NotationConfig | None = None) -> Path:
    """Render MusicXML (or MIDI) to PDF via the MuseScore CLI."""
    import subprocess

    config = config or NotationConfig()
    musicxml_path = Path(musicxml_path)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    exe = find_musescore(config.musescore_command)
    if exe is None:
        raise ExternalToolError(
            f"MuseScore CLI not found on PATH (tried '{config.musescore_command}' and "
            f"{', '.join(MUSESCORE_CANDIDATES)}). Install MuseScore 4, or pass "
            "--musescore-command / set NotationConfig.musescore_command to its executable. "
            "The MusicXML file was still written and can be opened in any notation editor."
        )
    cmd = [exe, "-o", str(pdf_path), str(musicxml_path)]
    log.info("Rendering PDF: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ExternalToolError(f"MuseScore failed ({proc.returncode}): {proc.stderr.strip()[:500]}")
    return pdf_path


def notate_drums(transcription: Transcription, out_dir: str | Path, config: NotationConfig | None = None, quantize: QuantizeConfig | None = None) -> dict[str, Path]:
    """Full Stage 4: MusicXML always, PDF if MuseScore is available and requested."""
    config = config or NotationConfig()
    out_dir = Path(out_dir)
    results: dict[str, Path] = {}

    musicxml = transcription_to_musicxml(transcription, out_dir / "drums.musicxml", config, quantize)
    results["musicxml"] = musicxml

    if config.render_pdf:
        try:
            results["pdf"] = render_pdf(musicxml, out_dir / "drums.pdf", config)
            log.info("Wrote drum sheet PDF: %s", results["pdf"])
        except ExternalToolError as exc:
            log.warning("PDF rendering skipped: %s", exc)
    return results
