"""Tests for the ground-truth groove bank (generation, scoring, evaluation)."""

from __future__ import annotations

import pytest

from drum_extractor.bank import (
    DEFAULT_TOLERANCE_S,
    PRESETS,
    _match_count,
    build_bank,
    evaluate_bank,
    format_report,
    humanize,
    score_hits,
)
from drum_extractor.config import DrumTranscriptionConfig
from drum_extractor.events import DrumHit
from drum_extractor.gm_drum_map import CANONICAL_INSTRUMENTS


# --- Matcher / scorer -----------------------------------------------------------------

def test_perfect_match_scores_one():
    ref = [DrumHit(0.0, "kick"), DrumHit(0.5, "snare"), DrumHit(1.0, "kick")]
    scores = score_hits(ref, list(ref))
    assert scores["overall"].f == 1.0
    assert scores["kick"].f == 1.0 and scores["snare"].f == 1.0


def test_misses_and_extras_scored():
    ref = [DrumHit(0.0, "kick"), DrumHit(1.0, "kick")]
    est = [DrumHit(0.0, "kick"), DrumHit(2.0, "kick"), DrumHit(3.0, "kick")]
    s = score_hits(ref, est)["kick"]
    assert s.tp == 1
    assert s.recall == 0.5  # one of two ref hits found
    assert s.precision == pytest.approx(1 / 3)  # two estimates are spurious


def test_tolerance_boundary():
    assert _match_count([1.0], [1.0 + DEFAULT_TOLERANCE_S - 0.001], DEFAULT_TOLERANCE_S) == 1
    assert _match_count([1.0], [1.0 + DEFAULT_TOLERANCE_S + 0.01], DEFAULT_TOLERANCE_S) == 0


def test_match_is_one_to_one():
    # Two estimates near one reference: only one may count (no double credit).
    assert _match_count([1.0], [0.99, 1.01], 0.05) == 1


def test_family_grouping():
    ref = [DrumHit(0.0, "hihat_closed")]
    est = [DrumHit(0.0, "hihat_open")]  # different articulation, same family
    assert score_hits(ref, est)["hihat"].f == 1.0


# --- Generators / humanize ------------------------------------------------------------

def test_all_presets_generate_valid_hits():
    for name, gen in PRESETS.items():
        hits, bpm = gen(bars=2)
        assert hits and bpm > 0, name
        for h in hits:
            assert h.instrument in CANONICAL_INSTRUMENTS, name
            assert h.time >= 0 and h.bar is not None and h.beat is not None, name


def test_humanize_preserves_labels_and_count():
    hits, _ = PRESETS["rock_8ths"](bars=2)
    jittered = humanize(hits, jitter_ms=8.0, vel_jitter=10, seed=1)
    assert len(jittered) == len(hits)
    assert all(h.time >= 0 for h in jittered)
    assert all(h.bar is not None and h.beat is not None for h in jittered)
    # Determinism: same seed -> same output.
    assert [h.time for h in humanize(hits, 8.0, 10, seed=1)] == [h.time for h in jittered]


# --- Build + evaluate (needs audio deps) ----------------------------------------------

def test_build_and_eval_roundtrip(tmp_path):
    pytest.importorskip("librosa")
    pytest.importorskip("soundfile")
    pytest.importorskip("pretty_midi")

    items = build_bank(tmp_path, presets=["rock_8ths"], bars=2, jitter_ms=4.0, seed=0)
    assert len(items) == 1
    item = items[0].path
    assert (item / "drums.wav").exists()
    assert (item / "reference.mid").exists()
    assert (item / "reference.json").exists()

    report = evaluate_bank(tmp_path, DrumTranscriptionConfig(backend="onset"))
    assert "rock_8ths" in report["items"]
    overall = report["aggregate"]["overall"]
    assert 0.0 <= overall["f"] <= 1.0
    # The synthesized rock groove is the easiest possible input; even the crude
    # onset fallback must find a meaningful fraction of the hits.
    assert overall["recall"] > 0.3

    table = format_report(report)
    assert "rock_8ths" in table and "AGGREGATE" in table


def test_eval_empty_bank_raises(tmp_path):
    with pytest.raises(ValueError):
        evaluate_bank(tmp_path)


def test_unknown_preset_rejected(tmp_path):
    with pytest.raises(ValueError):
        build_bank(tmp_path, presets=["rock_8ths", "polka_2step"])


def test_cli_bank_flags_parse():
    from drum_extractor.cli import build_parser

    args = build_parser().parse_args(["bank-build", "--presets", "rock_8ths,blast_beat", "--jitter-ms", "5"])
    assert args.command == "bank-build"
    args = build_parser().parse_args(["bank-eval", "bank", "--backend", "onset", "--tolerance-ms", "40"])
    assert args.tolerance_ms == 40.0
