"""Ground-truth groove bank: synthesized drum audio with known transcriptions.

Why a *bank* rather than raw training data: training-scale ADT corpora already
exist (ADTOF ~359h of rock/metal charts, E-GMD 444h) and a personal-scale
dataset can't compete with them for model training. What personal-scale data
IS uniquely good for:

1. **Measurement** — run our own transcriber on audio whose true hits are known
   and compute real precision/recall/F-scores per drum. Without this, every
   quantization/booster/classifier tweak is tuned blind.
2. **Accuracy regression testing** — CI can assert F-scores don't drop.
3. **Fine-tuning pairs later** — every item is an (audio, ground-truth MIDI)
   pair in exactly the format model fine-tuning needs, so nothing is wasted if
   we ever go there.

The synthesis trick (same as E-GMD / STAR-Drums): generate grooves
programmatically, render them with our own drum synthesizer (:mod:`sonify`),
then transcribe the rendered audio and score against the generator's truth.
Fully automatic and deterministic — no manual transcription required.

Bank layout::

    bank/
      rock_8ths/
        drums.wav          rendered audio (optionally humanized + noise)
        reference.mid      ground-truth MIDI
        reference.json     ground-truth Transcription (hits + tempo + meter)
        reference.musicxml ground truth through our notation stage (optional)
        meta.json
      punk_dbeat/
      ...
      report.json          written by `drum-extractor bank-eval`
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import DrumTranscriptionConfig
from .events import DrumHit, Transcription
from .gm_drum_map import FAMILY
from .logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_TOLERANCE_S = 0.05  # standard ADT evaluation window (mir_eval convention)


# --- Groove generators ----------------------------------------------------------------
# Each returns (hits, bpm). Hits carry exact bar/beat labels (the generator knows
# them), so the same data drives reference notation and quantization checks.

def _hit(bar: int, beat: float, instr: str, spb: float, bpb: int = 4, vel: int = 96) -> DrumHit:
    return DrumHit(time=((bar - 1) * bpb + beat) * spb, instrument=instr, velocity=vel, bar=bar, beat=beat)


def _groove_rock(bars: int = 4) -> tuple[list[DrumHit], float]:
    """Straight-8ths rock: hats 8ths, kick 1&3, snare 2&4, crash on the downbeat."""
    bpm = 120.0
    spb = 60.0 / bpm
    hits = [_hit(1, 0.0, "crash", spb)]
    for bar in range(1, bars + 1):
        hits += [_hit(bar, i * 0.5, "hihat_closed", spb, vel=80) for i in range(8)]
        hits += [_hit(bar, 0.0, "kick", spb), _hit(bar, 2.0, "kick", spb)]
        hits += [_hit(bar, 1.0, "snare", spb), _hit(bar, 3.0, "snare", spb)]
    return hits, bpm


def _groove_dbeat(bars: int = 4) -> tuple[list[DrumHit], float]:
    """Punk d-beat skeleton at 180 BPM."""
    bpm = 180.0
    spb = 60.0 / bpm
    hits = []
    for bar in range(1, bars + 1):
        hits += [_hit(bar, i * 0.5, "hihat_closed", spb, vel=80) for i in range(8)]
        hits += [_hit(bar, 0.0, "kick", spb), _hit(bar, 2.5, "kick", spb)]
        hits += [_hit(bar, 1.0, "snare", spb), _hit(bar, 3.0, "snare", spb)]
    return hits, bpm


def _groove_double_kick(bars: int = 4) -> tuple[list[DrumHit], float]:
    """200 BPM sixteenth-note double-kick under a backbeat — the metal stress test."""
    bpm = 200.0
    spb = 60.0 / bpm
    hits = []
    for bar in range(1, bars + 1):
        hits += [_hit(bar, i * 0.25, "kick", spb) for i in range(16)]
        hits += [_hit(bar, 1.0, "snare", spb), _hit(bar, 3.0, "snare", spb)]
        hits += [_hit(bar, float(b), "ride", spb, vel=84) for b in range(4)]
    return hits, bpm


def _groove_blast(bars: int = 4) -> tuple[list[DrumHit], float]:
    """Traditional blast at 220 BPM: kick and snare alternating 8ths, hat with snare."""
    bpm = 220.0
    spb = 60.0 / bpm
    hits = [_hit(1, 0.0, "crash", spb)]
    for bar in range(1, bars + 1):
        hits += [_hit(bar, i * 0.5, "kick", spb) for i in range(8)]
        for i in range(8):
            hits += [
                _hit(bar, i * 0.5 + 0.25, "snare", spb),
                _hit(bar, i * 0.5 + 0.25, "hihat_closed", spb, vel=76),
            ]
    return hits, bpm


def _groove_gallop(bars: int = 4) -> tuple[list[DrumHit], float]:
    """160 BPM kick gallops (8th + two 16ths per beat) with backbeat and ride."""
    bpm = 160.0
    spb = 60.0 / bpm
    hits = []
    for bar in range(1, bars + 1):
        for b in range(4):
            hits += [
                _hit(bar, b + 0.0, "kick", spb),
                _hit(bar, b + 0.5, "kick", spb),
                _hit(bar, b + 0.75, "kick", spb),
            ]
        hits += [_hit(bar, 1.0, "snare", spb), _hit(bar, 3.0, "snare", spb)]
        hits += [_hit(bar, float(b), "ride", spb, vel=84) for b in range(4)]
    return hits, bpm


def _groove_tom_fill(bars: int = 4) -> tuple[list[DrumHit], float]:
    """Rock groove with a final-bar 16th-note fill descending around the kit."""
    bpm = 120.0
    spb = 60.0 / bpm
    hits = []
    for bar in range(1, bars):
        hits += [_hit(bar, i * 0.5, "hihat_closed", spb, vel=80) for i in range(8)]
        hits += [_hit(bar, 0.0, "kick", spb), _hit(bar, 2.0, "kick", spb)]
        hits += [_hit(bar, 1.0, "snare", spb), _hit(bar, 3.0, "snare", spb)]
    fill = ["snare", "tom_high", "tom_mid", "tom_low"]
    for b, instr in enumerate(fill):
        hits += [_hit(bars, b + i * 0.25, instr, spb) for i in range(4)]
    return hits, bpm


def _groove_halftime_ghost(bars: int = 4) -> tuple[list[DrumHit], float]:
    """90 BPM half-time with ghost-note snares (low velocity)."""
    bpm = 90.0
    spb = 60.0 / bpm
    hits = []
    for bar in range(1, bars + 1):
        hits += [_hit(bar, i * 0.5, "hihat_closed", spb, vel=80) for i in range(8)]
        hits += [_hit(bar, 0.0, "kick", spb), _hit(bar, 2.0, "snare", spb)]
        hits += [
            _hit(bar, 1.25, "snare", spb, vel=30),
            _hit(bar, 2.75, "snare", spb, vel=30),
        ]
    return hits, bpm


def _groove_sparse(bars: int = 4) -> tuple[list[DrumHit], float]:
    """Slow, sparse quarter-note groove at 70 BPM."""
    bpm = 70.0
    spb = 60.0 / bpm
    hits = []
    for bar in range(1, bars + 1):
        hits += [_hit(bar, float(b), "hihat_closed", spb, vel=80) for b in range(4)]
        hits += [_hit(bar, 0.0, "kick", spb), _hit(bar, 2.0, "snare", spb)]
    return hits, bpm


PRESETS = {
    "rock_8ths": _groove_rock,
    "punk_dbeat": _groove_dbeat,
    "double_kick_16ths": _groove_double_kick,
    "blast_beat": _groove_blast,
    "metal_gallop": _groove_gallop,
    "tom_fill": _groove_tom_fill,
    "halftime_ghost": _groove_halftime_ghost,
    "sparse_slow": _groove_sparse,
}


# --- Humanization & rendering ---------------------------------------------------------

def humanize(hits: list[DrumHit], jitter_ms: float, vel_jitter: int = 0, seed: int = 0) -> list[DrumHit]:
    """Copy hits with timing/velocity jitter (bar/beat truth labels preserved)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for h in hits:
        t = max(0.0, h.time + float(rng.normal(0.0, jitter_ms / 1000.0)))
        v = int(min(127, max(1, h.velocity + (rng.integers(-vel_jitter, vel_jitter + 1) if vel_jitter else 0))))
        out.append(DrumHit(time=t, instrument=h.instrument, velocity=v, bar=h.bar, beat=h.beat))
    out.sort(key=lambda h: h.time)
    return out


def _add_noise(wav_path: Path, snr_db: float, seed: int = 0) -> None:
    """Mix white noise into a rendered stem at the given SNR (simulates bleed)."""
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(str(wav_path))
    rng = np.random.default_rng(seed)
    sig_rms = float(np.sqrt(np.mean(y**2))) or 1e-9
    noise = rng.standard_normal(y.shape) * sig_rms / (10 ** (snr_db / 20.0))
    mixed = y + noise
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.98:
        mixed *= 0.98 / peak
    sf.write(str(wav_path), mixed.astype("float32"), sr)


@dataclass
class BankItem:
    name: str
    path: Path
    bpm: float
    n_hits: int


def build_bank(
    out_dir: str | Path,
    presets: list[str] | None = None,
    bars: int = 4,
    jitter_ms: float = 0.0,
    vel_jitter: int = 0,
    noise_snr_db: float | None = None,
    seed: int = 0,
    write_notation: bool = False,
) -> list[BankItem]:
    """Generate and render the groove bank. Returns the items written."""
    if bars < 1:
        raise ValueError(f"bars must be >= 1 (got {bars})")
    if jitter_ms < 0:
        raise ValueError(f"jitter-ms must be >= 0 (got {jitter_ms})")
    if vel_jitter < 0:
        raise ValueError(f"vel-jitter must be >= 0 (got {vel_jitter})")

    out_dir = Path(out_dir)
    names = presets or list(PRESETS)
    items: list[BankItem] = []
    for name in names:
        if name not in PRESETS:
            raise ValueError(f"Unknown preset {name!r}; available: {', '.join(PRESETS)}")
        hits, bpm = PRESETS[name](bars=bars)
        truth = humanize(hits, jitter_ms, vel_jitter, seed=seed) if (jitter_ms or vel_jitter) else hits

        item_dir = out_dir / name
        item_dir.mkdir(parents=True, exist_ok=True)
        try:
            _write_bank_item(item_dir, name, truth, hits, bpm, bars, jitter_ms,
                             vel_jitter, noise_snr_db, seed, write_notation)
        except Exception:
            # A half-written item (reference.json without drums.wav) poisons
            # every later bank-eval of this directory — remove it on failure.
            import shutil

            shutil.rmtree(item_dir, ignore_errors=True)
            raise
        items.append(BankItem(name=name, path=item_dir, bpm=bpm, n_hits=len(truth)))
        log.info("Bank item %-18s %.0f BPM, %d hits -> %s", name, bpm, len(truth), item_dir)
    return items


def _write_bank_item(item_dir: Path, name: str, truth, hits, bpm: float, bars: int,
                     jitter_ms: float, vel_jitter: int, noise_snr_db: float | None,
                     seed: int, write_notation: bool) -> None:
    from .midi_io import write_drum_midi
    from .sonify import sonify_drums

    transcription = Transcription(drum_hits=truth, tempo=bpm, time_signature=(4, 4))
    (item_dir / "reference.json").write_text(json.dumps(transcription.to_dict(), indent=2))
    write_drum_midi(truth, item_dir / "reference.mid", tempo=bpm)
    sonify_drums(truth, item_dir / "drums.wav")
    if noise_snr_db is not None:
        _add_noise(item_dir / "drums.wav", noise_snr_db, seed=seed)
    if write_notation:
        from .config import NotationConfig, QuantizeConfig
        from .notation import notate_drums

        # The pristine grid hits (exact bar/beat) are the reference chart.
        ref = Transcription(drum_hits=hits, tempo=bpm, time_signature=(4, 4))
        notate_drums(ref, item_dir, NotationConfig(title=f"Reference: {name}", render_pdf=False), QuantizeConfig())
        # replace(), not rename(): overwrites an existing reference on re-build
        # (rename raises FileExistsError on Windows).
        (item_dir / "drums.musicxml").replace(item_dir / "reference.musicxml")
    (item_dir / "meta.json").write_text(
        json.dumps(
            {"preset": name, "bpm": bpm, "bars": bars, "jitter_ms": jitter_ms,
             "vel_jitter": vel_jitter, "noise_snr_db": noise_snr_db, "seed": seed},
            indent=2,
        )
    )


# --- Scoring --------------------------------------------------------------------------

@dataclass
class FamilyScore:
    tp: int = 0
    n_ref: int = 0
    n_est: int = 0

    @property
    def precision(self) -> float:
        return self.tp / self.n_est if self.n_est else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_ref if self.n_ref else 0.0

    @property
    def f(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict:
        return {"tp": self.tp, "n_ref": self.n_ref, "n_est": self.n_est,
                "precision": round(self.precision, 4), "recall": round(self.recall, 4), "f": round(self.f, 4)}


def _match_count(ref: list[float], est: list[float], tol: float) -> int:
    """Greedy two-pointer onset matching (each ref matches at most one est).

    All three branches share ONE difference value: recomputing ``ref[i] - tol``
    in the discard test rounds differently from ``abs(ref[i]-est[j])`` and could
    discard a reference that had an exact match sitting right there.
    """
    ref, est = sorted(ref), sorted(est)
    i = j = tp = 0
    while i < len(ref) and j < len(est):
        d = ref[i] - est[j]
        if abs(d) <= tol:
            tp += 1
            i += 1
            j += 1
        elif d > tol:  # estimate too early for this reference
            j += 1
        else:  # estimate too late — this reference is unmatched
            i += 1
    return tp


def score_hits(
    reference: list[DrumHit], estimated: list[DrumHit], tolerance: float = DEFAULT_TOLERANCE_S
) -> dict[str, FamilyScore]:
    """Score estimated hits against ground truth, per instrument family + overall.

    A hit counts as correct when the family matches and the onset falls within
    ``tolerance`` seconds (default ±50 ms, the standard ADT window).
    """
    fams = sorted({FAMILY.get(h.instrument, "other") for h in reference}
                  | {FAMILY.get(h.instrument, "other") for h in estimated})
    scores: dict[str, FamilyScore] = {}
    overall = FamilyScore()
    for fam in fams:
        ref_t = [h.time for h in reference if FAMILY.get(h.instrument, "other") == fam]
        est_t = [h.time for h in estimated if FAMILY.get(h.instrument, "other") == fam]
        s = FamilyScore(tp=_match_count(ref_t, est_t, tolerance), n_ref=len(ref_t), n_est=len(est_t))
        scores[fam] = s
        overall.tp += s.tp
        overall.n_ref += s.n_ref
        overall.n_est += s.n_est
    scores["overall"] = overall
    return scores


def evaluate_bank(
    bank_dir: str | Path,
    config: DrumTranscriptionConfig | None = None,
    tolerance: float = DEFAULT_TOLERANCE_S,
) -> dict:
    """Transcribe every bank item and score against its reference.

    Returns ``{"items": {name: {family: score...}}, "aggregate": {family: score...}}``
    and writes nothing — the CLI layer persists the report.
    """
    from .drums import transcribe_drums

    config = config or DrumTranscriptionConfig()
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive (got {tolerance*1000:.0f} ms) — "
                         "a non-positive window matches nothing and scores everything 0")
    bank_dir = Path(bank_dir)
    item_dirs = sorted(d for d in bank_dir.iterdir() if (d / "reference.json").exists())
    if not item_dirs:
        raise ValueError(f"No bank items under {bank_dir} (run `drum-extractor bank-build` first)")

    items: dict[str, dict] = {}
    agg: dict[str, FamilyScore] = {}
    for d in item_dirs:
        ref = Transcription.from_dict(json.loads((d / "reference.json").read_text()))
        est = transcribe_drums(d / "drums.wav", config)
        scores = score_hits(ref.drum_hits, est, tolerance)
        items[d.name] = {fam: s.to_dict() for fam, s in scores.items()}
        # Carry the item's generation settings so the report reads standalone.
        meta_path = d / "meta.json"
        if meta_path.exists():
            try:
                items[d.name]["meta"] = json.loads(meta_path.read_text())
            except (ValueError, OSError):
                pass
        for fam, s in scores.items():
            if fam == "overall":
                continue
            a = agg.setdefault(fam, FamilyScore())
            a.tp += s.tp
            a.n_ref += s.n_ref
            a.n_est += s.n_est
    total = FamilyScore(
        tp=sum(s.tp for s in agg.values()),
        n_ref=sum(s.n_ref for s in agg.values()),
        n_est=sum(s.n_est for s in agg.values()),
    )
    aggregate = {fam: s.to_dict() for fam, s in sorted(agg.items())}
    aggregate["overall"] = total.to_dict()
    return {"items": items, "aggregate": aggregate}


def format_report(report: dict) -> str:
    """Render an evaluation report as a fixed-width table."""
    fams = [f for f in report["aggregate"] if f != "overall"]
    header = f"{'item':<20}" + "".join(f"{f:>9}" for f in fams) + f"{'overall':>9}"
    lines = [header, "-" * len(header)]
    for name, scores in report["items"].items():
        row = f"{name:<20}"
        for f in fams:
            row += f"{scores[f]['f']:>9.2f}" if f in scores else f"{'-':>9}"
        row += f"{scores['overall']['f']:>9.2f}"
        lines.append(row)
    agg = report["aggregate"]
    row = f"{'AGGREGATE':<20}" + "".join(f"{agg[f]['f']:>9.2f}" for f in fams) + f"{agg['overall']['f']:>9.2f}"
    lines += ["-" * len(header), row]
    return "\n".join(lines)
