"""Regression tests for the full-application audit (2026-07).

Each test pins a specific defect found by the audit: if it fails, one of those
bugs came back. Grouped by module; heavier audio fixtures stay short (1-2 s).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from drum_extractor.config import (
    BassTranscriptionConfig,
    DrumTranscriptionConfig,
    GuitarTranscriptionConfig,
    NotationConfig,
    PipelineConfig,
    QuantizeConfig,
)
from drum_extractor.events import BassNote, DrumHit, Transcription


# --- tabs: chords must not vanish -------------------------------------------------------

GUITAR = GuitarTranscriptionConfig().tuning
BASS = BassTranscriptionConfig().tuning


def _chord(pitches, t=0.0):
    return [BassNote(start=t, end=t + 0.5, pitch=p) for p in pitches]


def test_duplicate_pitch_keeps_the_rest_of_the_chord():
    from drum_extractor.tabs import assign_frets

    notes = _chord([40, 40, 47, 52])  # doubled open E under a power chord
    unplaceable = assign_frets(notes, GUITAR, 24)
    placed = [n for n in notes if n.string is not None]
    assert len(placed) == 3, "only the duplicate should drop"
    assert unplaceable == 1
    assert sorted(n.pitch for n in placed) == [40, 47, 52]


def test_semitone_cluster_keeps_playable_subset():
    from drum_extractor.tabs import assign_frets

    notes = _chord([40, 41, 47])  # distortion smear: 41 makes a full voicing impossible
    assign_frets(notes, GUITAR, 24)
    placed = [n for n in notes if n.string is not None]
    assert len(placed) >= 2, "the playable 40+47 power chord must survive"


def test_shedding_removes_the_offending_middle_note():
    """Verification-round regression: when the MIDDLE note is unvoiceable,
    shedding only from the bottom used to discard a larger playable subset."""
    from drum_extractor.tabs import assign_frets

    notes = _chord([28, 34, 50])  # on a 3-string 24-fret bass, only {28, 50} voices
    assign_frets(notes, (28, 33, 38), 24)
    placed = sorted(n.pitch for n in notes if n.string is not None)
    assert placed == [28, 50], f"suboptimal shed kept {placed}"


def test_ascii_tab_renders_one_x_per_unplaceable_note():
    from drum_extractor.tabs import assign_frets, render_ascii_tab

    # 20 is below the bass fretboard; 28 lands on the open low E string.
    notes = [BassNote(0.0, 0.5, 20), BassNote(0.0, 0.5, 28)]
    assign_frets(notes, BASS, 24)
    tab = render_ascii_tab(notes, BASS)
    assert tab.count("x") == 1 and "0" in tab, "the x must not overwrite the fretted 0"

    both = [BassNote(0.0, 0.5, 20), BassNote(0.0, 0.5, 21)]
    assign_frets(both, BASS, 24)
    assert render_ascii_tab(both, BASS).count("x") == 2


def test_tab_header_names_tuning_and_tempo():
    from drum_extractor.tabs import render_ascii_tab

    notes = [BassNote(0.0, 0.5, 33)]
    from drum_extractor.tabs import assign_frets

    assign_frets(notes, BASS, 24)
    tab = render_ascii_tab(notes, BASS, title="songname", tempo=174.2)
    assert "songname" in tab
    assert "E1 A1 D2 G2" in tab  # tuning with octaves
    assert "174" in tab


# --- gp5: no silent note loss ------------------------------------------------------------

def test_gp5_same_slot_same_string_notes_both_survive(tmp_path):
    pytest.importorskip("guitarpro")
    import guitarpro as gp

    from drum_extractor.gp_export import write_gp5

    # Two real notes 50 ms apart on the same string: one 16th slot at 120 BPM.
    notes = [BassNote(0.00, 0.04, 28, string=0, fret=0),
             BassNote(0.05, 0.09, 30, string=0, fret=2)]
    out = write_gp5(notes, BASS, tmp_path / "b.gp5", tempo=120.0)
    song = gp.parse(str(out))
    got = [song.tracks[0].strings[n.string - 1].value + n.value
           for m in song.tracks[0].measures for b in m.voices[0].beats for n in b.notes]
    assert sorted(got) == [28, 30], "the second note must bump to the next slot, not vanish"


def test_gp5_negative_start_clamps_to_slot_zero(tmp_path):
    pytest.importorskip("guitarpro")
    import guitarpro as gp

    from drum_extractor.gp_export import write_gp5

    out = write_gp5([BassNote(-0.3, 0.1, 33, string=1, fret=0)], BASS, tmp_path / "n.gp5")
    song = gp.parse(str(out))
    n_notes = sum(len(b.notes) for m in song.tracks[0].measures for b in m.voices[0].beats)
    assert n_notes == 1, "a negative onset must clamp to the first slot, not disappear"


def test_gp5_carries_title_artist_and_meter(tmp_path):
    pytest.importorskip("guitarpro")
    import guitarpro as gp

    from drum_extractor.gp_export import write_gp5

    out = write_gp5([BassNote(0.0, 0.4, 33, string=1, fret=0)], BASS, tmp_path / "m.gp5",
                    title="My Song", artist="me", beats_per_bar=3)
    song = gp.parse(str(out))
    assert song.title == "My Song" and song.artist == "me"
    assert song.measureHeaders[0].timeSignature.numerator == 3


# --- drums: no phantom hits on drumless audio ---------------------------------------------

@pytest.fixture(scope="module")
def _wavs(tmp_path_factory):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    d = tmp_path_factory.mktemp("audit_wavs")
    sr = 44100
    t = np.arange(2 * sr) / sr
    sf.write(str(d / "tone.wav"), (0.5 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    rng = np.random.default_rng(0)
    sf.write(str(d / "quiet.wav"), (rng.standard_normal(2 * sr) * 1e-4).astype("float32"), sr)
    kick_t = np.linspace(0, 0.12, int(0.12 * sr))
    kick = (np.exp(-kick_t * 30) * np.sin(2 * np.pi * 60 * kick_t))
    kick[:300] += rng.standard_normal(300) * 0.15
    y = np.zeros(2 * sr, dtype="float32")
    for k in range(4):
        i = int((0.3 + k * 0.4) * sr)
        y[i:i + len(kick)] += kick.astype("float32")
    sf.write(str(d / "kicks.wav"), y, sr)
    return d


def test_sustained_tone_produces_no_hits(_wavs):
    pytest.importorskip("librosa")
    from drum_extractor.drums import _transcribe_onsets

    assert _transcribe_onsets(_wavs / "tone.wav", DrumTranscriptionConfig()) == []


def test_noise_floor_produces_no_hits(_wavs):
    pytest.importorskip("librosa")
    from drum_extractor.drums import _transcribe_onsets

    assert _transcribe_onsets(_wavs / "quiet.wav", DrumTranscriptionConfig()) == []


def test_real_kicks_still_detected(_wavs):
    pytest.importorskip("librosa")
    from drum_extractor.drums import _transcribe_onsets

    hits = _transcribe_onsets(_wavs / "kicks.wav", DrumTranscriptionConfig())
    for expect in (0.3, 0.7, 1.1, 1.5):
        assert any(abs(h.time - expect) < 0.1 for h in hits), f"kick at {expect}s missed"


def test_onset_text_skips_negative_times_and_reads_velocity(tmp_path):
    from drum_extractor.drums import _parse_onset_text

    p = tmp_path / "o.txt"
    p.write_text("-1.5,snare\nnan,kick\n0.5,kick,64\n1.0,hihat\n")
    hits = _parse_onset_text(p, velocity=96)
    assert [(h.time, h.instrument, h.velocity) for h in hits] == [
        (0.5, "kick", 64), (1.0, "hihat_closed", 96),
    ]


def test_onset_text_confidence_column_is_not_velocity(tmp_path):
    """Verification-round regression: a 0..1 confidence column must not clamp
    every hit to velocity 1 (a near-silent transcription)."""
    from drum_extractor.drums import _parse_onset_text

    p = tmp_path / "c.txt"
    p.write_text("0.5\t36\t0.93\n1.0\t38\t0.71\n")
    hits = _parse_onset_text(p, velocity=96)
    assert [h.velocity for h in hits] == [96, 96]


def test_quiet_but_real_drums_still_transcribe(tmp_path):
    """Verification-round regression: the silence gate must key on PEAK, not
    whole-track RMS — drums at -40 dBFS peak are quiet but fully detectable."""
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    from drum_extractor.drums import _transcribe_onsets

    sr = 44100
    y = np.zeros(3 * sr, dtype="float32")
    rng = np.random.default_rng(3)
    for k in range(6):
        i = int((0.25 + k * 0.4) * sr)
        y[i:i + 300] = (rng.standard_normal(300) * np.exp(-np.linspace(0, 6, 300))).astype("float32")
    y *= 10 ** (-40 / 20) / (np.max(np.abs(y)) + 1e-9)  # peak at -40 dBFS
    sf.write(str(tmp_path / "q.wav"), y, sr)
    hits = _transcribe_onsets(tmp_path / "q.wav", DrumTranscriptionConfig())
    assert len(hits) >= 4, f"quiet real drums gated to {len(hits)} hits"


def test_stale_empty_midi_does_not_shadow_txt_hits(tmp_path):
    """Verification-round regression: an empty .mid next to a valid .txt must
    not abort the candidate scan before the txt's hits are found."""
    from drum_extractor.drums import _read_adtof_output

    pm = pytest.importorskip("pretty_midi")
    empty = pm.PrettyMIDI()
    out_path = tmp_path / "song.adtof.mid"
    empty.write(str(out_path))
    (tmp_path / "song.txt").write_text("0.5\tkick\n1.0\tsnare\n")
    hits = _read_adtof_output(out_path, tmp_path / "song.wav", 96)
    assert len(hits) == 2


def test_unreadable_audio_raises_audio_load_error(tmp_path):
    pytest.importorskip("librosa")
    from drum_extractor.drums import _transcribe_onsets
    from drum_extractor.errors import AudioLoadError

    bad = tmp_path / "fake.wav"
    bad.write_text("not audio")
    with pytest.raises(AudioLoadError, match="fake.wav"):
        _transcribe_onsets(bad, DrumTranscriptionConfig())


# --- kick booster: one onset per kick ------------------------------------------------------

def _realistic_kicks(path, times, amps=None, sr=44100):
    """Pitch-swept kicks (the booster test suite's synth): one true transient each."""
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    total = int((max(times) + 0.4) * sr)
    audio = np.zeros(total)
    amps = amps or [1.0] * len(times)
    for t0, amp in zip(times, amps):
        i = int(t0 * sr)
        t = np.linspace(0, 0.06, int(0.06 * sr))
        freq = 120 * np.exp(-t * 22) + 45
        snd = amp * 0.9 * np.exp(-t * 35) * np.sin(2 * np.pi * np.cumsum(freq) / sr)
        j = min(i + len(snd), total)
        audio[i:j] += snd[: j - i]
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.9
    sf.write(str(path), audio.astype(np.float32), sr)


def test_isolated_kicks_no_universal_double_fire(tmp_path):
    """The audited bug: the 46 ms analysis window made EVERY kick fire twice
    (2x onsets, guaranteed). The short window fixes that; a residual trailing
    bump can still appear after SOME isolated kicks (a decaying kick body
    re-entering the low-pass band is indistinguishable from a soft stroke in
    band-limited analysis — quantize's slot dedupe absorbs it downstream), so
    the bound asserts the fix without over-promising: every kick found, and
    strictly fewer detections than the old 2-per-kick behavior."""
    pytest.importorskip("scipy")
    from drum_extractor.kick import detect_kick_onsets

    _realistic_kicks(tmp_path / "k.wav", [0.3, 0.7, 1.1, 1.5])
    onsets = detect_kick_onsets(tmp_path / "k.wav")
    for expect in (0.3, 0.7, 1.1, 1.5):
        assert any(abs(t - expect) < 0.04 for t in onsets), f"kick at {expect}s missed"
    assert len(onsets) < 8, f"universal double-fire is back: {onsets}"


def test_booster_on_complete_transcription_adds_few(tmp_path):
    pytest.importorskip("scipy")
    from drum_extractor.kick import boost_double_kick

    _realistic_kicks(tmp_path / "k.wav", [0.3, 0.7, 1.1, 1.5])
    hits = [DrumHit(t, "kick", 100) for t in (0.3, 0.7, 1.1, 1.5)]
    out = boost_double_kick(hits, tmp_path / "k.wav")
    added = len([h for h in out if h.instrument == "kick"]) - 4
    assert added <= 2, f"old behavior added a phantom per kick; got {added} for 4 kicks"


def test_soft_strokes_after_an_accent_survive(tmp_path):
    """Verification-round regression: an amplitude-based echo filter ate real
    soft double-kick strokes (~6 dB under a neighboring accent)."""
    pytest.importorskip("scipy")
    from drum_extractor.kick import detect_kick_onsets

    times = [0.3 + 0.07 * k for k in range(8)]
    amps = [1.0 if k % 4 == 0 else 0.5 for k in range(8)]  # accent + 3 soft
    _realistic_kicks(tmp_path / "g.wav", times, amps)
    onsets = detect_kick_onsets(tmp_path / "g.wav", min_gap_ms=30)
    assert len(onsets) >= 7, f"soft gallop strokes eaten: {len(onsets)}/8"


def test_kick_detector_clamps_tiny_analysis_sr(tmp_path):
    pytest.importorskip("scipy")
    from drum_extractor.kick import detect_kick_onsets

    _realistic_kicks(tmp_path / "k.wav", [0.3])
    detect_kick_onsets(tmp_path / "k.wav", sr=200)  # must not crash in butter()


# --- sonify: time validation ----------------------------------------------------------------

def test_sonify_drops_negative_and_nan_times(tmp_path):
    pytest.importorskip("soundfile")
    import soundfile as sf

    from drum_extractor.sonify import sonify_drums

    hits = [DrumHit(-0.5, "kick", 100), DrumHit(float("nan"), "snare", 90), DrumHit(0.2, "snare", 90)]
    out = sonify_drums(hits, tmp_path / "s.wav")
    y, sr = sf.read(str(out))
    assert abs(len(y) / sr - 1.2) < 0.05, "duration must come from the valid hit only"
    # Nothing may render in the final quarter (the negative hit used to wrap there).
    assert float(abs(y[-int(0.25 * sr):]).max()) < 1e-6


def test_sonify_rejects_all_invalid_and_absurd_times(tmp_path):
    pytest.importorskip("soundfile")
    from drum_extractor.sonify import sonify_drums

    with pytest.raises(ValueError):
        sonify_drums([DrumHit(-2.0, "kick", 100)], tmp_path / "x.wav")
    with pytest.raises(ValueError, match="Refusing"):
        sonify_drums([DrumHit(50_000.0, "kick", 100)], tmp_path / "y.wav")
    # t=0 with tail=0 must not crash on a zero-length buffer.
    sonify_drums([DrumHit(0.0, "kick", 100)], tmp_path / "z.wav", tail=0.0)


def test_gp5_clamps_zero_beats_per_bar(tmp_path):
    pytest.importorskip("guitarpro")
    import guitarpro as gp

    from drum_extractor.gp_export import write_gp5

    out = write_gp5([BassNote(0.0, 0.4, 33, string=1, fret=0)], BASS, tmp_path / "z.gp5",
                    beats_per_bar=0)
    song = gp.parse(str(out))
    assert song.measureHeaders[0].timeSignature.numerator == 1, "invalid 0/4 meter written"


# --- ensemble ---------------------------------------------------------------------------------

def test_empty_stem_raises_domain_error(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from drum_extractor.ensemble import average_stems
    from drum_extractor.errors import ExternalToolError

    empty = tmp_path / "e.wav"
    sf.write(str(empty), np.zeros((0, 1), dtype="float32"), 44100)
    with pytest.raises(ExternalToolError, match="empty"):
        average_stems([empty, empty], tmp_path / "out.wav")


def test_alignment_does_not_wrap_the_tail_to_the_front(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from drum_extractor.ensemble import average_stems

    sr = 44100
    n = sr // 2
    a = np.zeros((n, 1), dtype="float32")
    b = np.zeros((n, 1), dtype="float32")
    burst = (np.random.default_rng(0).standard_normal(400) * 0.5).astype("float32")
    a[3000:3400, 0] = burst
    b[2200:2600, 0] = burst          # b leads a by 800 samples
    b[-800:, 0] = 0.7                # loud tail that np.roll used to wrap to t=0
    sf.write(str(tmp_path / "a.wav"), a, sr)
    sf.write(str(tmp_path / "b.wav"), b, sr)
    out = average_stems([tmp_path / "a.wav", tmp_path / "b.wav"], tmp_path / "mix.wav")
    y, _ = sf.read(str(out), always_2d=True)
    assert float(abs(y[:800]).max()) < 0.05, "b's tail wrapped around to the start"


# --- notation ----------------------------------------------------------------------------------

def test_accent_survives_simultaneous_hit(tmp_path):
    pytest.importorskip("music21")
    from drum_extractor.notation import transcription_to_musicxml

    tr = Transcription(drum_hits=[
        DrumHit(0.0, "snare", velocity=120, bar=1, beat=0.0),
        DrumHit(0.0, "hihat_closed", velocity=80, bar=1, beat=0.0),
    ], tempo=120.0)
    xml = transcription_to_musicxml(tr, tmp_path / "a.musicxml", NotationConfig(), QuantizeConfig()).read_text()
    assert "<accent" in xml, "accented snare under a hi-hat lost its accent"


def test_metronome_mark_references_the_beat_unit(tmp_path):
    pytest.importorskip("music21")
    from drum_extractor.notation import transcription_to_musicxml

    tr = Transcription(drum_hits=[DrumHit(0.0, "kick", 100, bar=1, beat=0.0)],
                       tempo=178.0, time_signature=(6, 8))
    q = QuantizeConfig(beats_per_bar=6, beat_unit=8)
    xml = transcription_to_musicxml(tr, tmp_path / "b.musicxml", NotationConfig(), q).read_text()
    assert "<beat-unit>eighth</beat-unit>" in xml, "6/8 tempo printed as quarter = N (2x wrong)"


def test_duplicate_same_instrument_hits_dedupe(tmp_path):
    pytest.importorskip("music21")
    import re

    from drum_extractor.notation import transcription_to_musicxml

    tr = Transcription(drum_hits=[
        DrumHit(0.0, "snare", 110, bar=1, beat=0.0),
        DrumHit(0.0, "snare", 40, bar=1, beat=0.0),
    ], tempo=120.0)
    xml = transcription_to_musicxml(tr, tmp_path / "c.musicxml", NotationConfig(), QuantizeConfig()).read_text()
    assert len(re.findall(r"<unpitched>", xml)) == 1, "doubled notehead on the same staff position"


# --- quantize ------------------------------------------------------------------------------------

def test_slightly_early_onset_does_not_create_empty_leading_bar(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    from drum_extractor.quantize import quantize

    sr = 44100
    y = np.zeros(10 * sr, dtype="float32")
    rng = np.random.default_rng(0)
    for k in range(16):
        i = int((0.5 + k * 0.5) * sr)
        y[i:i + 400] = (rng.standard_normal(400) * np.exp(-np.linspace(0, 8, 400))).astype("float32")
    sf.write(str(tmp_path / "click.wav"), y, sr)

    hits = []
    for bar in range(4):
        b0 = 0.5 + bar * 2.0
        first = b0 - (0.008 if bar == 0 else 0.0)  # 8 ms early, the audited trigger
        hits += [DrumHit(first, "kick", 100), DrumHit(b0 + 0.5, "snare", 100),
                 DrumHit(b0 + 1.0, "kick", 100), DrumHit(b0 + 1.5, "snare", 100)]
    tr = Transcription(drum_hits=hits)
    quantize(tr, tmp_path / "click.wav",
             QuantizeConfig(backend="librosa", grid=16, grid_mode="constant", fixed_tempo=120.0))
    assert tr.drum_hits[0].bar == 1, "an 8ms-early kick shifted every bar number"
    assert tr.downbeats[0] >= 0.0, "negative downbeats leaked into the transcription"


def test_quantize_config_validates_grid():
    with pytest.raises(ValueError, match="grid"):
        QuantizeConfig(grid=0)
    with pytest.raises(ValueError, match="grid"):
        QuantizeConfig(grid=-8)
    with pytest.raises(ValueError, match="min_bpm"):
        QuantizeConfig(min_bpm=-10)


# --- events / IR ------------------------------------------------------------------------------------

def test_transcription_json_is_versioned_and_forward_compatible():
    tr = Transcription(drum_hits=[DrumHit(0.0, "kick", 100)],
                       guitar_notes=[BassNote(0.0, 0.5, 45, string=1, fret=0)])
    d = tr.to_dict()
    assert d["format"] == "drum-extractor/transcription" and d["version"] == 1

    d["drum_hits"][0]["flam"] = True  # a field from "the future"
    d["something_new"] = {"x": 1}
    back = Transcription.from_dict(d)
    assert back.drum_hits[0].instrument == "kick"
    assert back.guitar_notes[0].pitch == 45


# --- midi_io / gm map ----------------------------------------------------------------------------------

def test_aux_percussion_maps_to_sensible_voices():
    from drum_extractor.gm_drum_map import gm_to_canonical

    assert gm_to_canonical(56) == "ride_bell"   # cowbell
    assert gm_to_canonical(54) == "hihat_closed"  # tambourine
    assert gm_to_canonical(39) == "snare"       # hand clap


# --- pipeline -----------------------------------------------------------------------------------------

def _stub_stem(path: Path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    sr = 44100
    y = np.zeros(2 * sr, dtype="float32")
    rng = np.random.default_rng(1)
    for k in range(4):
        i = int((0.25 + k * 0.5) * sr)
        y[i:i + 300] = (rng.standard_normal(300) * 0.4).astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)


def test_missing_input_leaves_no_output_dir(tmp_path):
    from drum_extractor.errors import AudioLoadError
    from drum_extractor.pipeline import run_pipeline

    with pytest.raises(AudioLoadError):
        run_pipeline(tmp_path / "nope.mp3", PipelineConfig(output_dir=tmp_path / "out"))
    assert not (tmp_path / "out" / "nope").exists()


def test_rerun_without_bass_removes_stale_bass_outputs(tmp_path):
    pytest.importorskip("librosa")
    from drum_extractor.pipeline import run_pipeline

    song_dir = tmp_path / "out" / "song"
    _stub_stem(song_dir / "stems" / "drums.wav")
    for stale in ("bass.mid", "bass.tab.txt", "bass.gp5"):
        (song_dir / stale).write_bytes(b"old")

    config = PipelineConfig(
        output_dir=tmp_path / "out",
        drums=DrumTranscriptionConfig(backend="onset"),
        do_separation=False, do_bass_transcription=False,
        do_quantize=False, do_notation=False, do_sonify=False,
    )
    run_pipeline(tmp_path / "song.wav", config)  # input need not exist: stems reused
    for stale in ("bass.mid", "bass.tab.txt", "bass.gp5"):
        assert not (song_dir / stale).exists(), f"stale {stale} survived a --no-bass rerun"


def test_drum_part_abbreviation_is_not_perc(tmp_path):
    """Continuation systems print the part ABBREVIATION: music21's generic
    Percussion says 'Perc', which made system 2+ look like a separate
    percussion part instead of the same drum staff."""
    pytest.importorskip("music21")
    from drum_extractor.notation import transcription_to_musicxml

    tr = Transcription(drum_hits=[DrumHit(0.0, "kick", 100, bar=1, beat=0.0)], tempo=120.0)
    xml = transcription_to_musicxml(tr, tmp_path / "p.musicxml", NotationConfig(), QuantizeConfig()).read_text()
    assert "<part-abbreviation>Dr.</part-abbreviation>" in xml
    assert "Perc" not in xml


def test_visible_rests_in_exported_musicxml(tmp_path):
    """Verification-round regression: the unhide-rests pass must run on the
    EXPORTED file — music21 injects print-object='no' rests during export."""
    pytest.importorskip("music21")
    from drum_extractor.notation import transcription_to_musicxml

    tr = Transcription(drum_hits=[DrumHit(0.0, "snare", 100, bar=1, beat=1.0)], tempo=120.0)
    xml = transcription_to_musicxml(tr, tmp_path / "r.musicxml", NotationConfig(), QuantizeConfig()).read_text()
    assert 'print-object="no"' not in xml, "hidden padding rests leaked into the sheet"


def test_reused_config_does_not_leak_title_between_songs(tmp_path):
    pytest.importorskip("librosa")
    pytest.importorskip("music21")
    from drum_extractor.pipeline import run_pipeline

    config = PipelineConfig(
        output_dir=tmp_path / "out",
        drums=DrumTranscriptionConfig(backend="onset"),
        do_separation=False, do_bass_transcription=False,
        do_quantize=False, do_sonify=False,
    )
    config.notation.render_pdf = False
    for song in ("alpha", "beta"):
        _stub_stem(tmp_path / "out" / song / "stems" / "drums.wav")
        run_pipeline(tmp_path / f"{song}.wav", config)
    beta_xml = (tmp_path / "out" / "beta" / "drums.musicxml").read_text()
    assert "<work-title>beta</work-title>" in beta_xml, "first song's title leaked into the second"


def test_failed_bass_stage_preserves_previous_outputs(tmp_path, monkeypatch):
    """Verification-round regression: a transient bass failure must not delete
    the previous run's good bass files."""
    pytest.importorskip("librosa")
    import drum_extractor.pipeline as pl

    song_dir = tmp_path / "out" / "song"
    _stub_stem(song_dir / "stems" / "drums.wav")
    (song_dir / "stems" / "bass.wav").write_bytes(b"RIFFxxxx")
    for old in ("bass.mid", "bass.tab.txt"):
        (song_dir / old).write_bytes(b"good old output")

    def boom(*a, **k):
        raise RuntimeError("model download blocked")

    monkeypatch.setattr(pl, "_do_bass", boom)
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        drums=DrumTranscriptionConfig(backend="onset"),
        do_separation=False, do_quantize=False, do_notation=False, do_sonify=False,
    )
    result = pl.run_pipeline(tmp_path / "song.wav", config)
    assert any("bass transcription" in w for w in result.warnings)
    for old in ("bass.mid", "bass.tab.txt"):
        assert (song_dir / old).exists(), f"failed stage destroyed previous {old}"


def test_successful_empty_rerun_cleans_stale_outputs(tmp_path, monkeypatch):
    pytest.importorskip("librosa")
    import drum_extractor.pipeline as pl

    song_dir = tmp_path / "out" / "song"
    _stub_stem(song_dir / "stems" / "drums.wav")
    (song_dir / "stems" / "bass.wav").write_bytes(b"RIFFxxxx")
    (song_dir / "bass.tab.txt").write_bytes(b"stale")

    monkeypatch.setattr(pl, "_do_bass", lambda result, config, song_dir: None)  # ran fine, found nothing
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        drums=DrumTranscriptionConfig(backend="onset"),
        do_separation=False, do_quantize=False, do_notation=False, do_sonify=False,
    )
    pl.run_pipeline(tmp_path / "song.wav", config)
    assert not (song_dir / "bass.tab.txt").exists(), "stale tab survived a successful empty rerun"


def test_summary_lists_transcription_json_and_pdf_skip_warning(tmp_path, monkeypatch):
    pytest.importorskip("librosa")
    pytest.importorskip("music21")
    import shutil as _shutil

    from drum_extractor.pipeline import run_pipeline

    monkeypatch.setattr(_shutil, "which", lambda *a, **k: None)  # no MuseScore anywhere
    song_dir = tmp_path / "out" / "song"
    _stub_stem(song_dir / "stems" / "drums.wav")
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        drums=DrumTranscriptionConfig(backend="onset"),
        do_separation=False, do_bass_transcription=False,
        do_quantize=False, do_sonify=False,
    )
    result = run_pipeline(tmp_path / "song.wav", config)
    text = result.summary()
    assert "transcription" in text and "transcription.json" in text
    assert any("PDF skipped" in w for w in result.warnings)
    # And the sheet is titled after the song, not the generic default.
    assert "<work-title>song</work-title>" in result.musicxml.read_text()


# --- bank ----------------------------------------------------------------------------------------------

def test_bank_build_validates_inputs(tmp_path):
    from drum_extractor.bank import build_bank

    with pytest.raises(ValueError, match="bars"):
        build_bank(tmp_path / "b", presets=["sparse_slow"], bars=0)
    with pytest.raises(ValueError, match="jitter"):
        build_bank(tmp_path / "b", presets=["sparse_slow"], jitter_ms=-1)
    with pytest.raises(ValueError, match="vel-jitter"):
        build_bank(tmp_path / "b", presets=["sparse_slow"], vel_jitter=-2)


def test_failed_bank_item_is_cleaned_up(tmp_path, monkeypatch):
    pytest.importorskip("soundfile")
    import drum_extractor.bank as bank_mod
    from drum_extractor.bank import build_bank

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr("drum_extractor.sonify.sonify_drums", boom)
    with pytest.raises(RuntimeError):
        build_bank(tmp_path / "bank", presets=["sparse_slow"])
    assert not (tmp_path / "bank" / "sparse_slow").exists(), "poisoned partial item left behind"


def test_bank_eval_rejects_nonpositive_tolerance(tmp_path):
    from drum_extractor.bank import evaluate_bank

    with pytest.raises(ValueError, match="tolerance"):
        evaluate_bank(tmp_path, tolerance=-0.005)


# --- doctor / musescore parity ----------------------------------------------------------------------------

def test_doctor_and_renderer_share_musescore_lookup(monkeypatch):
    import shutil as _shutil

    from drum_extractor.notation import find_musescore

    monkeypatch.setattr(_shutil, "which", lambda name: "/usr/bin/musescore4" if name == "musescore4" else None)
    assert find_musescore() == "/usr/bin/musescore4"
    assert find_musescore("mscore") == "/usr/bin/musescore4", \
        "the configured-name miss must fall back to the candidate list"


def test_doctor_reports_guitarpro_check():
    from drum_extractor.doctor import run_doctor

    assert any("PyGuitarPro" in c.feature for c in run_doctor())


# --- web ------------------------------------------------------------------------------------------------

@pytest.fixture()
def web_app(tmp_path, monkeypatch):
    pytest.importorskip("flask")
    from drum_extractor.pipeline import PipelineResult
    from drum_extractor.web import server as web_server

    def fake_pipeline(audio_path, config, on_stage=None):
        stems_dir = Path(config.output_dir) / "stems"
        stems_dir.mkdir(parents=True, exist_ok=True)
        result = PipelineResult()
        p = stems_dir / "drums.wav"
        p.write_bytes(b"RIFFxxxx")
        result.stems.drums = p
        return result

    monkeypatch.setattr(web_server, "run_pipeline", fake_pipeline)
    factory = lambda job_id, guitar=False: PipelineConfig(output_dir=tmp_path / "jobs" / job_id)  # noqa: E731
    return web_server.create_app(config_factory=factory, output_dir=tmp_path, sync=True)


def test_upload_with_huge_filename_returns_clean_400(web_app):
    import io

    client = web_app.test_client()
    res = client.post("/upload", data={"song": (io.BytesIO(b"RIFF"), "a" * 5000 + ".wav")},
                      content_type="multipart/form-data")
    assert res.status_code == 200 or res.status_code == 400
    assert res.is_json, "raw HTML 500 page leaked to the client"


def test_mixer_title_is_json_encoded_against_injection(web_app):
    import io

    client = web_app.test_client()
    evil = "so`ng${window.x=1}"
    res = client.post("/upload", data={"song": (io.BytesIO(b"RIFF"), evil + ".wav")},
                      content_type="multipart/form-data")
    page = client.get(res.get_json()["mixer"]).data.decode()
    assert "const TITLE =" in page
    # The raw backtick/${} must never appear inside a JS template literal sink.
    assert "a.download = TITLE + '-custom-mix.wav'" in page
