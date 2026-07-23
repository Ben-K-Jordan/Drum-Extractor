"""Small logging helper so every module logs consistently under one namespace."""

from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``drum_extractor``."""
    short = name.split(".")[-1]
    return logging.getLogger(f"drum_extractor.{short}")


def configure_logging(verbose: bool = False) -> None:
    """Set up a single stream handler for the CLI. Idempotent.

    Library users who manage their own logging can simply never call this.
    """
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger("drum_extractor").setLevel(logging.DEBUG if verbose else logging.INFO)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s", "%H:%M:%S"))
    root = logging.getLogger("drum_extractor")
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.propagate = False
    _CONFIGURED = True
