"""Install-time wiring that fails silently if it drifts."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_adtof_extra_names_the_real_package():
    # pip rejects a direct-URL requirement whose name differs from the
    # package's metadata name, and the installers fall back to the onset
    # transcriber with only a warning, so a wrong name ships nobody ADTOF.
    extras = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]
    (spec,) = extras["adtof"]
    name, url = (part.strip() for part in spec.split("@", 1))
    assert name == "adtof-pytorch"
    assert re.fullmatch(r"git\+https://github\.com/xavriley/ADTOF-pytorch(@\S+)?", url)
