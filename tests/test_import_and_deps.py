"""The package must import with zero heavy deps, and missing deps must fail loudly."""

import pytest


def test_package_imports_without_heavy_deps():
    import drum_extractor

    assert drum_extractor.__version__
    # Public API is exposed.
    assert hasattr(drum_extractor, "run_pipeline")
    assert hasattr(drum_extractor, "PipelineConfig")


def test_separation_reports_missing_demucs_clearly(tmp_path):
    """If demucs isn't installed, we get an actionable MissingDependencyError."""
    pytest.importorskip  # keep import order tidy
    try:
        import demucs  # noqa: F401

        pytest.skip("demucs is installed; nothing to assert about its absence")
    except ModuleNotFoundError:
        pass

    from drum_extractor.config import SeparationConfig
    from drum_extractor.errors import MissingDependencyError
    from drum_extractor.separation import separate

    audio = tmp_path / "song.wav"
    audio.write_bytes(b"RIFF")  # existence check passes; demucs import fails first
    with pytest.raises(MissingDependencyError) as exc:
        separate(audio, tmp_path / "out", SeparationConfig(device="cpu"))
    assert "demucs" in str(exc.value)
    assert "pip install" in str(exc.value)


def test_cli_parser_builds():
    from drum_extractor.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["run", "song.mp3", "--grid", "32"])
    assert args.command == "run"
    assert args.grid == 32
