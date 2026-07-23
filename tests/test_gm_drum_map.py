from drum_extractor import gm_drum_map as gm


def test_canonical_gm_roundtrip():
    for name in gm.CANONICAL_INSTRUMENTS:
        note = gm.canonical_to_gm(name)
        # Every canonical instrument maps to a GM note that maps back to *a*
        # canonical name (aliases may collapse, e.g. crash1/crash2 -> crash).
        assert isinstance(note, int)
        assert gm.gm_to_canonical(note) in gm.CANONICAL_INSTRUMENTS


def test_normalize_accepts_varied_labels():
    assert gm.normalize_instrument("kick") == gm.KICK
    assert gm.normalize_instrument("BD") == gm.KICK
    assert gm.normalize_instrument("cymbals") == gm.CRASH
    assert gm.normalize_instrument("36") == gm.KICK  # GM note number as string
    assert gm.normalize_instrument("38") == gm.SNARE
    assert gm.normalize_instrument("hi-hat") == gm.HIHAT_CLOSED


def test_every_instrument_has_placement():
    for name in gm.CANONICAL_INSTRUMENTS:
        assert name in gm.PLACEMENT
        place = gm.PLACEMENT[name]
        assert place.voice in (1, 2)
        assert place.stem in ("up", "down")
        assert place.notehead in {"normal", "x", "circle-x", "diamond", "cross"}
