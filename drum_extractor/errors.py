"""Exception types shared across the pipeline."""

from __future__ import annotations


class DrumExtractorError(Exception):
    """Base class for all errors raised by drum_extractor."""


class MissingDependencyError(DrumExtractorError):
    """A stage was invoked but an optional dependency is not installed.

    The message always includes the exact ``pip install`` command that fixes it,
    so a partially-installed environment fails loudly and actionably instead of
    with a bare ``ModuleNotFoundError``.
    """

    def __init__(self, feature: str, package: str, extra: str | None = None) -> None:
        self.feature = feature
        self.package = package
        self.extra = extra
        hint = f'pip install "drum-extractor[{extra}]"' if extra else f"pip install {package}"
        super().__init__(
            f"{feature} needs the '{package}' package, which is not installed.\n"
            f"    Install it with:  {hint}"
        )


class ExternalToolError(DrumExtractorError):
    """An external command-line tool (e.g. MuseScore, ADTOF) failed or is missing."""


class AudioLoadError(DrumExtractorError):
    """An input audio file could not be read."""
