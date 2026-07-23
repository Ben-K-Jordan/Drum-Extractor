"""Tests for the environment checkup (`drum-extractor doctor`)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from drum_extractor.doctor import MISSING, OK, format_report, run_doctor

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_doctor_runs_and_covers_every_stage():
    checks = run_doctor()
    features = " ".join(c.feature for c in checks)
    for area in ("Python", "core:", "separation:", "drums:", "bass:", "quantize:", "notation:", "web:"):
        assert area in features, f"doctor has no check for {area}"
    # Python + core checks must be present and marked core.
    assert any(c.core for c in checks)


def test_doctor_report_includes_fix_hints():
    checks = run_doctor()
    report = format_report(checks)
    assert "drum-extractor doctor" in report
    # Every missing entry surfaces its fix command in the report.
    for c in checks:
        if c.status == MISSING and c.fix:
            assert c.fix in report
    # The verdict line is one of the three known forms.
    assert any(s in report for s in ("Everything is ready", "Usable.", "BROKEN:"))


def test_doctor_exit_code_reflects_core_health():
    from drum_extractor.doctor import doctor_main

    checks = run_doctor()
    core_broken = any(c.core and c.status == MISSING for c in checks)
    assert doctor_main() == (1 if core_broken else 0)


def test_python_dash_m_entrypoint():
    out = subprocess.run(
        [sys.executable, "-m", "drum_extractor", "--version"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert out.returncode == 0
    assert "drum-extractor" in out.stdout


def test_cli_doctor_parses():
    from drum_extractor.cli import build_parser

    args = build_parser().parse_args(["doctor"])
    assert args.command == "doctor"
