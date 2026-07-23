"""Phase 4 — sonify a drum transcription back to audio for by-ear checking.

The honest reality is that fast-metal transcription needs human verification.
The most practical correction aid short of a full GUI is to *hear* the
transcription: render each detected hit as a short synthetic drum sound, so you
can play it against the original stem and immediately catch missed or wrong hits.

Pure numpy + soundfile — no models, fully testable.
"""

from __future__ import annotations

from pathlib import Path

from .errors import MissingDependencyError
from .events import DrumHit
from .gm_drum_map import (
    CRASH, CHINA, HIHAT_CLOSED, HIHAT_OPEN, HIHAT_PEDAL, KICK, RIDE, RIDE_BELL,
    SNARE, SIDE_STICK, SPLASH, TOM_HIGH, TOM_LOW, TOM_MID,
)
from .logging_utils import get_logger

log = get_logger(__name__)

# Synthesis recipe per instrument. Two kinds:
#   ("tone",  fundamental_hz, decay)              — kick/toms: pitched sine w/ drop
#   ("noise", (band_lo, band_hi), decay)          — band-limited noise burst
#   ("snare", fundamental_hz, decay)              — body tone + mid noise + wires
# Bands are chosen to resemble REAL drum spectra, not just to sound okay: the
# groove bank scores the transcriber on this audio, so if (e.g.) the snare were
# plain white noise its energy would sit mostly above 5 kHz like a cymbal and
# any threshold tuned on it would be meaningless for real snares.
_VOICES = {
    KICK:         ("tone", 55, 30),
    TOM_LOW:      ("tone", 110, 18),
    TOM_MID:      ("tone", 160, 18),
    TOM_HIGH:     ("tone", 220, 18),
    SNARE:        ("snare", 190, 25),
    SIDE_STICK:   ("noise", (1500, 6000), 60),
    HIHAT_CLOSED: ("noise", (4000, 13000), 90),
    HIHAT_OPEN:   ("noise", (4000, 13000), 25),
    HIHAT_PEDAL:  ("noise", (3500, 9000), 70),
    RIDE:         ("noise", (3000, 10000), 12),
    RIDE_BELL:    ("noise", (4000, 9000), 14),
    CRASH:        ("noise", (2500, 14000), 8),
    CHINA:        ("noise", (2000, 12000), 8),
    SPLASH:       ("noise", (5000, 15000), 16),
}


def _band_noise(rng, length: int, sr: int, lo: float, hi: float):
    """Band-limited white noise via FFT masking (pure numpy)."""
    import numpy as np

    noise = rng.standard_normal(length)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(length, 1.0 / sr)
    spec[(freqs < lo) | (freqs > hi)] = 0.0
    out = np.fft.irfft(spec, n=length)
    peak = float(np.max(np.abs(out))) or 1.0
    return out / peak


def sonify_drums(hits: list[DrumHit], out_path: str | Path, sr: int = 44100, tail: float = 1.0) -> Path:
    """Render drum hits to a WAV of synthetic drum sounds aligned to their times."""
    try:
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Sonification", "soundfile", extra="drums") from exc

    if not hits:
        raise ValueError("No hits to sonify.")

    duration = max(h.time for h in hits) + tail
    n = int(duration * sr)
    buf = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(0)

    for h in hits:
        kind, param, decay = _VOICES.get(h.instrument, ("noise", (200, 8000), 30))
        length = int(min(0.4, 3.0 / decay) * sr)
        t = np.linspace(0, length / sr, length, endpoint=False)
        env = np.exp(-t * decay)
        if kind == "tone":
            fsweep = param * (1.5 * np.exp(-t * 20) + 1.0)  # slight pitch drop
            sig = np.sin(2 * np.pi * np.cumsum(fsweep) / sr)
        elif kind == "snare":
            # Real-snare balance: a low-mid body tone, mid-band shell noise, and
            # a smaller high "wires" component.
            sig = (
                0.45 * np.sin(2 * np.pi * param * t)
                + 0.60 * _band_noise(rng, length, sr, 150, 4500)
                + 0.30 * _band_noise(rng, length, sr, 5000, 11000)
            )
        else:
            lo, hi = param
            sig = _band_noise(rng, length, sr, lo, hi)
        amp = 0.2 + 0.6 * (h.velocity / 127.0)
        snd = (amp * env * sig).astype(np.float32)
        i = int(h.time * sr)
        j = min(i + length, n)
        buf[i:j] += snd[: j - i]

    peak = float(np.max(np.abs(buf))) or 1.0
    buf = (buf / peak * 0.9).astype(np.float32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), buf, sr)
    log.info("Sonified %d hits -> %s", len(hits), out_path)
    return out_path


def write_onset_csv(hits: list[DrumHit], out_path: str | Path) -> Path:
    """Write a ``time,instrument,velocity`` CSV for loading onsets into a DAW/editor."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["time,instrument,velocity"]
    for h in sorted(hits, key=lambda x: x.time):
        lines.append(f"{h.time:.4f},{h.instrument},{h.velocity}")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
