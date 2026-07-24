"""Shared tablature engine: polyphonic string/fret assignment + ASCII rendering.

Used by both the bass and guitar paths. The assigner handles CHORDS — notes
sounding together must land on distinct strings, without string crossings —
which the previous single-note Viterbi could not express. Bass gets double-stop
support out of the same code.

Algorithm: notes are grouped into chords (onsets within a small window), each
chord enumerates its feasible voicings (higher pitch on higher string — the
standard, crossing-free convention), and a DP across chords picks the voicing
sequence minimizing hand movement + fret span, with free repositioning after a
long rest.
"""

from __future__ import annotations

from math import inf

from .events import BassNote
from .logging_utils import get_logger

log = get_logger(__name__)

CHORD_EPS_S = 0.03  # onsets within 30 ms belong to one chord
REST_RESET_S = 1.5  # a rest this long lets the hand move for free
START_FRET = 5  # neutral starting hand position
OPEN_BIAS = 0.4  # slight bias against open strings mid-phrase
SPAN_WEIGHT = 1.0  # cost per fret of intra-chord stretch
MAX_SPAN = 5  # voicings stretching further are discarded
RESET_ANCHOR = 0.2  # gentle pull toward START_FRET after a long rest
MAX_VOICINGS = 24  # per-chord candidate cap (plenty for 6 strings)


def group_chords(notes: list[BassNote], eps: float = CHORD_EPS_S) -> list[list[BassNote]]:
    """Group time-sorted notes into chords by onset proximity."""
    groups: list[list[BassNote]] = []
    for n in sorted(notes, key=lambda x: (x.start, x.pitch)):
        if groups and n.start - groups[-1][0].start <= eps:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def _voicings(pitches: list[int], tuning: tuple[int, ...], frets: int) -> list[list[tuple[int, int]]]:
    """Feasible (string, fret) assignments for one chord's pitches (ascending).

    Monotonic: the i-th lowest pitch sits on a lower string than the (i+1)-th —
    no string crossings, the standard way chords are actually fingered.
    """
    out: list[list[tuple[int, int]]] = []

    def bt(i: int, next_string: int, current: list[tuple[int, int]]) -> None:
        if len(out) >= MAX_VOICINGS:
            return
        if i == len(pitches):
            fretted = [f for _, f in current if f > 0]
            if fretted and max(fretted) - min(fretted) > MAX_SPAN:
                return
            out.append(list(current))
            return
        for s in range(next_string, len(tuning)):
            f = pitches[i] - tuning[s]
            if 0 <= f <= frets:
                current.append((s, f))
                bt(i + 1, s + 1, current)
                current.pop()

    bt(0, 0, [])
    return out


def _hand_pos(voicing: list[tuple[int, int]], prev_pos: float) -> float:
    fretted = [f for _, f in voicing if f > 0]
    return float(min(fretted)) if fretted else prev_pos  # open chords leave the hand in place


def _voicing_cost(voicing: list[tuple[int, int]], prev_pos: float, resting: bool) -> tuple[float, float]:
    """(cost, new_hand_position) for playing ``voicing`` after ``prev_pos``."""
    pos = _hand_pos(voicing, prev_pos)
    fretted = [f for _, f in voicing if f > 0]
    span = (max(fretted) - min(fretted)) if fretted else 0
    move = RESET_ANCHOR * abs(pos - START_FRET) if resting else abs(pos - prev_pos)
    opens = sum(1 for _, f in voicing if f == 0)
    return move + SPAN_WEIGHT * span + OPEN_BIAS * opens, pos


def assign_frets(notes: list[BassNote], tuning: tuple[int, ...], frets: int) -> int:
    """Assign (string, fret) in place; returns how many notes were unplaceable.

    Unplaceable notes (outside the instrument's range, or chord notes beyond
    the string count) keep ``string``/``fret`` as ``None`` and are rendered as
    'x' in the tab while keeping their true pitch in the MIDI.
    """
    chords = group_chords(notes)
    unplaceable = 0

    # Split each chord into placeable pitches + hopeless leftovers.
    playable_chords: list[tuple[list[BassNote], list[list[tuple[int, int]]]]] = []
    for chord in chords:
        chord_sorted = sorted(chord, key=lambda n: n.pitch)
        in_range: list[BassNote] = []
        dropped: list[BassNote] = []
        for n in chord_sorted:
            if 0 <= n.pitch - tuning[0] and n.pitch - tuning[-1] <= frets:
                in_range.append(n)
            else:
                dropped.append(n)
        # Duplicate pitches (basic-pitch re-strike jitter) can never sit on
        # distinct strings without a crossing — keep one of each.
        seen: set[int] = set()
        uniq: list[BassNote] = []
        for n in in_range:
            if n.pitch in seen:
                dropped.append(n)
            else:
                seen.add(n.pitch)
                uniq.append(n)
        in_range = uniq
        # More notes than strings: keep the highest ones (melody survives).
        if len(in_range) > len(tuning):
            dropped += in_range[: len(in_range) - len(tuning)]
            in_range = in_range[len(in_range) - len(tuning):]
        voicings = _voicings([n.pitch for n in in_range], tuning, frets) if in_range else []
        # A cluster with no crossing-free voicing (semitone smears from
        # distorted chords) must not sink the whole chord. Shed ONE note at a
        # time — whichever single removal restores a voicing, trying the
        # lowest first (melody survives) — so an unvoiceable MIDDLE note
        # can't force away more notes than necessary.
        while in_range and not voicings:
            removal = 0
            for i in range(len(in_range)):
                trial = [n.pitch for j, n in enumerate(in_range) if j != i]
                if trial and _voicings(trial, tuning, frets):
                    removal = i
                    break
            dropped.append(in_range.pop(removal))
            if in_range:
                voicings = _voicings([n.pitch for n in in_range], tuning, frets)
        unplaceable += len(dropped)
        if in_range:
            playable_chords.append((in_range, voicings))

    if playable_chords:
        # DP over chords: state = chosen voicing (carrying its hand position).
        first_notes, first_voicings = playable_chords[0]
        states = []
        for v in first_voicings:
            cost, pos = _voicing_cost(v, START_FRET, resting=True)
            states.append((cost, pos))
        back: list[list[int]] = [[-1] * len(first_voicings)]

        for k in range(1, len(playable_chords)):
            prev_notes, prev_voicings = playable_chords[k - 1]
            cur_notes, cur_voicings = playable_chords[k]
            gap = cur_notes[0].start - max(n.end for n in prev_notes)
            resting = gap > REST_RESET_S
            new_states, new_back = [], []
            for v in cur_voicings:
                best_c, best_j = inf, 0
                for j, (pc, ppos) in enumerate(states):
                    c_step, _ = _voicing_cost(v, ppos, resting)
                    if pc + c_step < best_c:
                        best_c, best_j = pc + c_step, j
                _, npos = _voicing_cost(v, states[best_j][1], resting)
                new_states.append((best_c, npos))
                new_back.append(best_j)
            states, back = new_states, back + [new_back]

        j = min(range(len(states)), key=lambda i: states[i][0])
        for k in range(len(playable_chords) - 1, -1, -1):
            chord_notes, voicings = playable_chords[k]
            for note, (s, f) in zip(chord_notes, voicings[j]):
                note.string, note.fret = s, f
            j = back[k][j]

    if unplaceable:
        log.warning(
            "%d note(s) could not be placed (outside the fretboard, duplicated in a "
            "chord, or an unvoiceable cluster; tuning low=%d, %d frets, %d strings); "
            "marked 'x' in the tab, true pitch kept in the MIDI.",
            unplaceable, tuning[0], frets, len(tuning),
        )
    return unplaceable


# Tunings are conventionally spelled with flats (Eb standard, Db, drop Ab).
_NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def string_names(tuning: tuple[int, ...]) -> list[str]:
    return [_NOTE_NAMES[p % 12] for p in tuning]


def note_name(pitch: int) -> str:
    """MIDI pitch -> name with octave (28 -> 'E1')."""
    return f"{_NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def tab_header(tuning: tuple[int, ...], title: str | None = None, tempo: float | None = None) -> str:
    """Human-readable header for a tab file: song, tuning (with octaves), legend."""
    lines = []
    if title:
        lines.append(f"== {title} ==")
    info = f"tuning: {' '.join(note_name(p) for p in tuning)} (low to high)"
    if tempo:
        info += f"   tempo: ~{round(tempo)} BPM"
    lines.append(info)
    lines.append("x = note the transcriber heard but that doesn't fit the fretboard (kept in the MIDI)")
    return "\n".join(lines)


def render_ascii_tab(
    notes: list[BassNote],
    tuning: tuple[int, ...],
    width: int = 76,
    title: str | None = None,
    tempo: float | None = None,
) -> str:
    """Chord-aware ASCII tab, wrapped into multiple systems at ``width`` chars.

    Notes sounding together share one column; unplaceable notes render as 'x'
    (one per note, on free strings from the bottom up — when every string in a
    column already carries a fret, further x's have nowhere to draw and only
    the MIDI keeps them). ``title``/``tempo`` add a header block.
    """
    names = string_names(tuning)
    label_w = max((len(n) for n in names), default=1)

    columns: list[dict[int, str]] = []
    for chord in group_chords(notes):
        col: dict[int, str] = {}
        xs = 0
        for n in chord:
            if n.string is None or n.fret is None:
                xs += 1
            else:
                col[n.string] = str(n.fret)
        # One visible 'x' per unplaceable note, never overwriting a real fret.
        for s in range(len(tuning)):
            if xs == 0:
                break
            if s not in col:
                col[s] = "x"
                xs -= 1
        columns.append(col)

    # Wrap columns into systems.
    systems: list[list[dict[int, str]]] = []
    current: list[dict[int, str]] = []
    used = 0
    body_width = max(width - (label_w + 2), 12)
    for col in columns:
        w = max(max((len(t) for t in col.values()), default=1) + 1, 3)
        if current and used + w > body_width:
            systems.append(current)
            current, used = [], 0
        current.append(col)
        used += w
    if current or not systems:
        systems.append(current)

    lines: list[str] = []
    if title or tempo:
        lines.append(tab_header(tuning, title, tempo))
        lines.append("")
    for si, system in enumerate(systems):
        if si:
            lines.append("")
        for s in range(len(tuning) - 1, -1, -1):  # highest string on top
            row = []
            for col in system:
                w = max(max((len(t) for t in col.values()), default=1) + 1, 3)
                token = col.get(s, "")
                row.append(token.rjust(w - 1, "-") + "-" if token else "-" * w)
            body = "".join(row) or "-" * 8
            lines.append(f"{names[s].rjust(label_w)}|-{body}")
    return "\n".join(lines)
