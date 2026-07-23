"""General MIDI percussion mapping and drum-notation placement.

Two things live here:

1. ``CANONICAL_TO_GM`` — maps the pipeline's internal instrument names to
   General MIDI note numbers on channel 10. Transcribers (ADTOF, etc.) emit a
   small class set; we normalise to these canonical names and then to GM so
   every downstream stage (MIDI export, notation) speaks the same language.

2. ``PLACEMENT`` — where each instrument sits on a 5-line drum staff and how it
   is drawn (notehead + which voice/stem direction). Positions follow the common
   drum-set convention used by MuseScore and most method books. There is no
   single ISO standard for drum notation, so treat this as a sensible default
   you can tweak in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Canonical instrument vocabulary -------------------------------------------------

KICK = "kick"
SNARE = "snare"
SIDE_STICK = "side_stick"
HIHAT_CLOSED = "hihat_closed"
HIHAT_OPEN = "hihat_open"
HIHAT_PEDAL = "hihat_pedal"
TOM_HIGH = "tom_high"
TOM_MID = "tom_mid"
TOM_LOW = "tom_low"
CRASH = "crash"
RIDE = "ride"
RIDE_BELL = "ride_bell"
CHINA = "china"
SPLASH = "splash"

CANONICAL_INSTRUMENTS = (
    KICK, SNARE, SIDE_STICK,
    HIHAT_CLOSED, HIHAT_OPEN, HIHAT_PEDAL,
    TOM_HIGH, TOM_MID, TOM_LOW,
    CRASH, RIDE, RIDE_BELL, CHINA, SPLASH,
)

# Canonical name -> General MIDI note number (channel 10).
CANONICAL_TO_GM: dict[str, int] = {
    KICK: 36,          # Bass Drum 1
    SNARE: 38,         # Acoustic Snare
    SIDE_STICK: 37,    # Side Stick
    HIHAT_CLOSED: 42,  # Closed Hi-Hat
    HIHAT_OPEN: 46,    # Open Hi-Hat
    HIHAT_PEDAL: 44,   # Pedal Hi-Hat
    TOM_HIGH: 50,      # High Tom
    TOM_MID: 47,       # Low-Mid Tom
    TOM_LOW: 43,       # High Floor Tom
    CRASH: 49,         # Crash Cymbal 1
    RIDE: 51,          # Ride Cymbal 1
    RIDE_BELL: 53,     # Ride Bell
    CHINA: 52,         # Chinese Cymbal
    SPLASH: 55,        # Splash Cymbal
}

# Reverse map: every GM drum note we might encounter -> canonical name.
# Includes common aliases (e.g. both crash cymbals collapse to "crash").
GM_TO_CANONICAL: dict[int, str] = {
    35: KICK, 36: KICK,
    37: SIDE_STICK,
    38: SNARE, 40: SNARE,
    42: HIHAT_CLOSED,
    44: HIHAT_PEDAL,
    46: HIHAT_OPEN,
    41: TOM_LOW, 43: TOM_LOW, 45: TOM_MID, 47: TOM_MID, 48: TOM_HIGH, 50: TOM_HIGH,
    49: CRASH, 57: CRASH,
    51: RIDE, 59: RIDE,
    52: CHINA,
    53: RIDE_BELL,
    55: SPLASH,
}

# The 5-class vocabulary most ADT models (ADTOF) emit -> canonical name.
# Used when normalising a transcriber's coarse output.
ADT_5CLASS_TO_CANONICAL: dict[str, str] = {
    "kick": KICK, "bass_drum": KICK, "bd": KICK, "0": KICK,
    "snare": SNARE, "sd": SNARE, "1": SNARE,
    "hihat": HIHAT_CLOSED, "hi-hat": HIHAT_CLOSED, "hh": HIHAT_CLOSED, "2": HIHAT_CLOSED,
    "toms": TOM_MID, "tom": TOM_MID, "tt": TOM_MID, "3": TOM_MID,
    "cymbals": CRASH, "cymbal": CRASH, "cy": CRASH, "4": CRASH,
}


@dataclass(frozen=True)
class Placement:
    """How one drum voice is drawn on the staff.

    ``step``/``octave`` are treble-clef reference positions that fix the vertical
    line/space; music21 uses them as ``displayStep``/``displayOctave`` on an
    unpitched note. ``notehead`` is a music21 notehead name. ``voice`` 1 is the
    hands (stems up), voice 2 is the feet (stems down).
    """

    step: str
    octave: int
    notehead: str
    stem: str  # "up" or "down"
    voice: int  # 1 = hands, 2 = feet


# Conventional drum-set staff placement. Edit here to re-map your own kit.
PLACEMENT: dict[str, Placement] = {
    KICK:         Placement("F", 4, "normal",   "down", 2),
    SNARE:        Placement("C", 5, "normal",   "up",   1),
    SIDE_STICK:   Placement("C", 5, "x",        "up",   1),
    TOM_LOW:      Placement("A", 4, "normal",   "up",   1),
    TOM_MID:      Placement("D", 5, "normal",   "up",   1),
    TOM_HIGH:     Placement("E", 5, "normal",   "up",   1),
    HIHAT_CLOSED: Placement("G", 5, "x",        "up",   1),
    HIHAT_OPEN:   Placement("G", 5, "circle-x", "up",   1),
    HIHAT_PEDAL:  Placement("D", 4, "x",        "down", 2),
    RIDE:         Placement("F", 5, "x",        "up",   1),
    RIDE_BELL:    Placement("F", 5, "diamond",  "up",   1),
    CRASH:        Placement("A", 5, "x",        "up",   1),
    CHINA:        Placement("B", 5, "x",        "up",   1),
    SPLASH:       Placement("B", 5, "x",        "up",   1),
}

# Human-readable labels, handy for text output / debugging.
DISPLAY_NAMES: dict[str, str] = {
    KICK: "Kick", SNARE: "Snare", SIDE_STICK: "Side stick",
    HIHAT_CLOSED: "Hi-hat (closed)", HIHAT_OPEN: "Hi-hat (open)", HIHAT_PEDAL: "Hi-hat (pedal)",
    TOM_HIGH: "Tom (high)", TOM_MID: "Tom (mid)", TOM_LOW: "Tom (floor)",
    CRASH: "Crash", RIDE: "Ride", RIDE_BELL: "Ride bell", CHINA: "China", SPLASH: "Splash",
}


def canonical_to_gm(name: str) -> int:
    """Canonical instrument name -> GM note number. Falls back to snare (38)."""
    return CANONICAL_TO_GM.get(name, 38)


def gm_to_canonical(note: int) -> str:
    """GM note number -> canonical instrument name. Falls back to snare."""
    return GM_TO_CANONICAL.get(note, SNARE)


def normalize_instrument(label: str) -> str:
    """Best-effort map from an arbitrary transcriber label to a canonical name.

    Accepts canonical names, GM note numbers (as ``str`` or ``int``), and the
    coarse 5-class ADT vocabulary.
    """
    key = str(label).strip().lower()
    if key in CANONICAL_INSTRUMENTS:
        return key
    if key in ADT_5CLASS_TO_CANONICAL:
        return ADT_5CLASS_TO_CANONICAL[key]
    if key.isdigit():
        return gm_to_canonical(int(key))
    return ADT_5CLASS_TO_CANONICAL.get(key, SNARE)
