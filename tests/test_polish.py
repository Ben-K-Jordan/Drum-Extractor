"""Coverage the pipeline/CLI auditor flagged as missing: CLI-level end-to-end,
doctor fix-hint accuracy against real extras, and the bank noise path."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _make_drum_stem(path: Path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    sr = 44100
    y = np.zeros(3 * sr, dtype="float32")
    rng = np.random.default_rng(7)
    for k in range(6):
        i = int((0.25 + k * 0.5) * sr)
        y[i:i + 300] = (rng.standard_normal(300) * 0.4).astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)


def test_cli_run_end_to_end_with_reused_stems(tmp_path, capsys):
    pytest.importorskip("librosa")
    pytest.importorskip("music21")
    from drum_extractor.cli import main

    _make_drum_stem(tmp_path / "out" / "song" / "stems" / "drums.wav")
    rc = main([
        "run", str(tmp_path / "song.wav"), "-o", str(tmp_path / "out"),
        "--reuse-stems", "--drum-backend", "onset", "--no-bass", "--no-pdf",
    ])
    out, err = capsys.readouterr()
    assert rc == 0
    assert out.count("Pipeline results:") == 1, "summary printed more than once"
    assert "transcription.json" in out
    assert (tmp_path / "out" / "song" / "drums.musicxml").exists()
    # Sheet is titled after the song.
    assert "<work-title>song</work-title>" in (tmp_path / "out" / "song" / "drums.musicxml").read_text()


def test_cli_missing_input_is_a_one_liner(tmp_path, caplog):
    from drum_extractor.cli import main

    with caplog.at_level("ERROR"):
        rc = main(["run", str(tmp_path / "nope.mp3"), "-o", str(tmp_path / "out")])
    assert rc == 1
    assert any("Input audio not found" in r.message for r in caplog.records)


def test_doctor_fix_hints_name_real_extras():
    from drum_extractor.doctor import run_doctor

    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    for check in run_doctor():
        for extra in re.findall(r'drum-extractor\[(\w[\w-]*)\]', check.fix):
            assert extra in extras, f"doctor advertises nonexistent extra [{extra}] for {check.feature!r}"


def test_bank_noise_path_produces_evaluable_items(tmp_path):
    pytest.importorskip("librosa")
    from drum_extractor.bank import build_bank, evaluate_bank
    from drum_extractor.config import DrumTranscriptionConfig

    build_bank(tmp_path / "bank", presets=["rock_8ths"], noise_snr_db=20.0)
    report = evaluate_bank(tmp_path / "bank", DrumTranscriptionConfig(backend="onset"))
    f = report["aggregate"]["overall"]["f"]
    assert f > 0.4, f"noisy bank item scored implausibly low: {f}"
    assert report["items"]["rock_8ths"]["meta"]["noise_snr_db"] == 20.0
