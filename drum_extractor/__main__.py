"""Allow `python -m drum_extractor ...` (equivalent to the drum-extractor CLI)."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
