"""Flask app behind ``drum-extractor web``.

Flow (matching the intended UX):
1. **Drop page** (``/``) — drag a song/recording in.
2. Upload starts a background pipeline job (separating ALL four stems so the
   mixer can rebalance the whole song, plus drum/bass transcription+notation).
3. **Mixer page** (``/mixer/<id>``) — polls job status, then loads the stems
   into the Web Audio API: one fader per stem, mute/solo, master volume.
   The volume-adjusted mix is rendered *in the browser* (OfflineAudioContext),
   so downloading it needs no server round-trip.
4. Download buttons for the drum sheet (PDF if MuseScore rendered it, else
   MusicXML), drum MIDI, and bass tab.

Security posture: this is a LOCAL personal tool (binds 127.0.0.1 by default).
Job ids are server-generated UUIDs and every served file path comes from the
job's own results — nothing is resolved from user-supplied paths.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..config import PipelineConfig, SeparationConfig
from ..errors import MissingDependencyError
from ..logging_utils import get_logger
from ..pipeline import PipelineResult, run_pipeline

log = get_logger(__name__)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif"}
MIXER_STEMS = ("drums", "bass", "other", "vocals")
MIXER_STEMS_6S = ("drums", "bass", "guitar", "piano", "other", "vocals")


def default_config_factory(output_dir: Path, model: str = "htdemucs_ft", device: str = "auto"):
    """Build the per-job pipeline config used by the web UI."""

    def make(job_id: str, guitar: bool = False) -> PipelineConfig:
        # Guitar tabs need the 6-stem model's guitar stem; the mixer then also
        # gets Guitar and Keys channels.
        job_model = model
        if guitar and model in ("htdemucs", "htdemucs_ft"):
            job_model = "htdemucs_6s"
        return PipelineConfig(
            output_dir=output_dir / "jobs" / job_id,
            separation=SeparationConfig(model=job_model, device=device, stems=MIXER_STEMS_6S if guitar else MIXER_STEMS),
            do_guitar_transcription=guitar,
            do_sonify=False,  # not useful in the web flow; saves time
        )

    return make


@dataclass
class Job:
    id: str
    title: str
    state: str = "queued"  # queued | <stage name> | done | error
    error: str | None = None
    result: PipelineResult | None = None
    timing: dict | None = None  # tempo / downbeats / drum hits for the mixer overlays
    created: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = {"id": self.id, "title": self.title, "state": self.state, "error": self.error}
        if self.result is not None:
            d["stems"] = {name: f"/stems/{self.id}/{name}" for name in self.result.stems.available()}
            downloads = {}
            if self.result.pdf:
                downloads["sheet"] = {"url": f"/download/{self.id}/sheet", "label": "Drum sheet (PDF)"}
            elif self.result.musicxml:
                downloads["sheet"] = {"url": f"/download/{self.id}/sheet", "label": "Drum sheet (MusicXML)"}
            if self.result.drum_midi:
                downloads["midi"] = {"url": f"/download/{self.id}/midi", "label": "Drum MIDI"}
            if self.result.bass_tab:
                downloads["tab"] = {"url": f"/download/{self.id}/tab", "label": "Bass tab"}
            if self.result.bass_midi:
                downloads["bmidi"] = {"url": f"/download/{self.id}/bmidi", "label": "Bass MIDI"}
            if self.result.bass_gp:
                downloads["gpb"] = {"url": f"/download/{self.id}/gpb", "label": "Bass tab (.gp5)"}
            if self.result.guitar_tab:
                downloads["gtab"] = {"url": f"/download/{self.id}/gtab", "label": "Guitar tab"}
            if self.result.guitar_gp:
                downloads["gpg"] = {"url": f"/download/{self.id}/gpg", "label": "Guitar tab (.gp5)"}
            if self.result.guitar_midi:
                downloads["gmidi"] = {"url": f"/download/{self.id}/gmidi", "label": "Guitar MIDI"}
            d["downloads"] = downloads
            d["warnings"] = self.result.warnings
            if self.timing:
                d["timing"] = self.timing
            # Inline sheet preview: SVG via verovio when available, else the PDF.
            if self.result.musicxml and _verovio_available():
                d["sheet_view"] = {"kind": "svg", "url": f"/sheet/{self.id}.svg"}
            elif self.result.pdf:
                d["sheet_view"] = {"kind": "pdf", "url": f"/sheet/{self.id}.pdf"}
        return d


_ARTIFACT_FIELDS = (
    "musicxml", "pdf", "drum_midi", "bass_midi", "bass_tab",
    "guitar_midi", "guitar_tab", "bass_gp", "guitar_gp",
)


def _timing_from(result: PipelineResult) -> dict | None:
    """Extract what the mixer overlays need: tempo, downbeats, hit markers."""
    from ..gm_drum_map import FAMILY

    t = result.transcription
    if not (t.tempo or t.downbeats or t.drum_hits):
        return None
    return {
        "tempo": t.tempo,
        "downbeats": [round(x, 4) for x in t.downbeats[:4000]],
        "drum_hits": [[round(h.time, 3), FAMILY.get(h.instrument, "other")] for h in t.drum_hits[:8000]],
    }


def _persist_job(job: Job, job_dir: Path) -> None:
    """Write a manifest so processed songs survive a server restart."""
    import json

    r = job.result
    if r is None:
        return
    manifest = {
        "id": job.id,
        "title": job.title,
        "created": job.created,
        "stems": {k: str(v) for k, v in r.stems.available().items()},
        "artifacts": {f: str(getattr(r, f)) for f in _ARTIFACT_FIELDS if getattr(r, f)},
        "warnings": r.warnings,
        "timing": job.timing,
    }
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(json.dumps(manifest, indent=2))
    except OSError as exc:
        log.warning("Could not persist job manifest (%s) — the mixer won't survive a restart.", exc)


def _restore_jobs(store: "JobStore", output_dir: Path) -> int:
    """Rebuild done jobs from manifests written by earlier runs."""
    import json

    jobs_root = output_dir / "jobs"
    if not jobs_root.is_dir():
        return 0
    restored = 0
    for manifest_path in sorted(jobs_root.glob("*/job.json")):
        try:
            m = json.loads(manifest_path.read_text())
            result = PipelineResult()
            for name, p in m.get("stems", {}).items():
                if Path(p).exists() and hasattr(result.stems, name):
                    setattr(result.stems, name, Path(p))
            for f, p in m.get("artifacts", {}).items():
                if f in _ARTIFACT_FIELDS and Path(p).exists():
                    setattr(result, f, Path(p))
            result.warnings = list(m.get("warnings", []))
            if not result.stems.available():
                continue  # stems were deleted; nothing to mix
            job = Job(
                id=str(m["id"]), title=str(m.get("title", "song")), state="done",
                result=result, timing=m.get("timing"), created=float(m.get("created", 0)),
            )
            store.put(job)
            restored += 1
        except Exception as exc:  # one bad manifest must not break startup
            log.warning("Skipping unreadable job manifest %s (%s)", manifest_path, exc)
    if restored:
        log.info("Restored %d processed song(s) from previous runs", restored)
    return restored


# verovio's toolkit only finds its font resources when CONSTRUCTED on the main
# thread, but Flask handlers run on worker threads — so one shared toolkit is
# built eagerly in create_app() (main thread) and reused under a lock (the
# toolkit is stateful and not thread-safe).
_VEROVIO: dict = {"tk": None, "lock": threading.Lock()}


def _init_verovio() -> None:
    if _VEROVIO["tk"] is not None:
        return
    try:
        import verovio

        _VEROVIO["tk"] = verovio.toolkit()
        log.info("verovio ready — inline sheet previews enabled")
    except Exception as exc:
        log.info("verovio unavailable (%s) — sheet preview falls back to PDF/downloads", exc)


def _verovio_available() -> bool:
    return _VEROVIO["tk"] is not None


def render_sheet_svg(musicxml_path: Path, max_pages: int = 20) -> str | None:
    """Engrave MusicXML to stacked SVG pages with verovio. None on failure."""
    tk = _VEROVIO["tk"]
    if tk is None:
        return None
    try:
        with _VEROVIO["lock"]:
            tk.setOptions({"scale": 38, "adjustPageHeight": True, "pageWidth": 1900})
            if not tk.loadData(musicxml_path.read_text()):
                return None
            pages = min(tk.getPageCount(), max_pages)
            return "\n".join(tk.renderToSVG(p) for p in range(1, pages + 1))
    except Exception as exc:  # engraving must never break the mixer
        log.warning("verovio rendering failed: %s", exc)
        return None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, title: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], title=title)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def put(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent_done(self, limit: int = 12) -> list[Job]:
        with self._lock:
            done = [j for j in self._jobs.values() if j.state == "done"]
        return sorted(done, key=lambda j: j.created, reverse=True)[:limit]


def _run_job(job: Job, audio_path: Path, config: PipelineConfig) -> None:
    try:
        def on_stage(stage: str) -> None:
            job.state = stage

        result = run_pipeline(audio_path, config, on_stage=on_stage)
        job.result = result
        if not result.stems.available():
            job.state = "error"
            job.error = "Separation produced no stems — see the server log."
        else:
            job.timing = _timing_from(result)
            job.state = "done"
            _persist_job(job, Path(config.output_dir))
    except Exception as exc:  # surfaced via /job/<id>, not a crashed thread
        log.exception("Job %s failed", job.id)
        job.state = "error"
        job.error = str(exc)


def create_app(config_factory=None, output_dir: str | Path = "output", sync: bool = False):
    """Create the Flask app.

    ``config_factory(job_id) -> PipelineConfig`` builds each job's config
    (defaults to :func:`default_config_factory`). ``sync=True`` runs jobs
    inline instead of in a thread — used by the tests for determinism.
    """
    try:
        from flask import Flask, abort, jsonify, render_template, request, send_file
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Web UI", "flask", extra="web") from exc

    output_dir = Path(output_dir)
    make_config = config_factory or default_config_factory(output_dir)
    _init_verovio()  # must happen on the main thread (see note above)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # accept long WAV recordings
    store = JobStore()
    app.extensions["drum_extractor_jobs"] = store  # handy for tests
    _restore_jobs(store, output_dir)  # processed songs survive restarts

    @app.get("/")
    def index():
        recent = [
            {"title": j.title, "mixer": f"/mixer/{j.id}"}
            for j in store.recent_done()
        ]
        return render_template("index.html", formats=", ".join(sorted(ALLOWED_EXTENSIONS)), recent=recent)

    @app.post("/upload")
    def upload():
        file = request.files.get("song")
        if file is None or not file.filename:
            return jsonify({"error": "No file received."}), 400
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type {suffix or '(none)'}. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

        job = store.create(title=Path(file.filename).stem)
        upload_dir = output_dir / "jobs" / job.id
        upload_dir.mkdir(parents=True, exist_ok=True)
        # Server-controlled filename: keep only the sanitized stem + suffix,
        # capped well under the 255-byte filesystem limit.
        safe_stem = "".join(c for c in Path(file.filename).stem if c.isalnum() or c in "-_ ").strip()[:120] or "song"
        audio_path = upload_dir / f"{safe_stem}{suffix}"
        try:
            file.save(str(audio_path))
        except OSError as exc:
            # Don't leave a zombie 'queued' job whose mixer page spins forever.
            job.state = "error"
            job.error = f"Could not save the upload: {exc.strerror or exc}"
            shutil.rmtree(upload_dir, ignore_errors=True)
            return jsonify({"error": job.error}), 400

        want_guitar = request.form.get("guitar") in ("1", "true", "on")
        try:
            config = make_config(job.id, guitar=want_guitar)
        except TypeError:  # custom factories that predate the guitar option
            config = make_config(job.id)
        if sync:
            _run_job(job, audio_path, config)
        else:
            threading.Thread(target=_run_job, args=(job, audio_path, config), daemon=True).start()
        return jsonify({"id": job.id, "mixer": f"/mixer/{job.id}"})

    @app.get("/job/<job_id>")
    def job_status(job_id: str):
        job = store.get(job_id)
        if job is None:
            abort(404)
        return jsonify(job.to_dict())

    @app.get("/mixer/<job_id>")
    def mixer(job_id: str):
        job = store.get(job_id)
        if job is None:
            abort(404)
        return render_template("mixer.html", job_id=job.id, title=job.title)

    @app.get("/stems/<job_id>/<stem>")
    def stem_audio(job_id: str, stem: str):
        job = store.get(job_id)
        if job is None or job.result is None:
            abort(404)
        path = job.result.stems.available().get(stem)  # allowlist: only real stem names resolve
        if path is None or not Path(path).exists():
            abort(404)
        return send_file(Path(path).resolve(), mimetype="audio/wav", conditional=True)

    @app.get("/sheet/<job_id>.svg")
    def sheet_svg(job_id: str):
        job = store.get(job_id)
        if job is None or job.result is None or not job.result.musicxml:
            abort(404)
        svg = render_sheet_svg(Path(job.result.musicxml))
        if svg is None:
            abort(404)
        return app.response_class(svg, mimetype="image/svg+xml")

    @app.get("/sheet/<job_id>.pdf")
    def sheet_pdf(job_id: str):
        job = store.get(job_id)
        if job is None or job.result is None or not job.result.pdf:
            abort(404)
        return send_file(Path(job.result.pdf).resolve(), mimetype="application/pdf", conditional=True)

    @app.get("/download/<job_id>/<kind>")
    def download(job_id: str, kind: str):
        job = store.get(job_id)
        if job is None or job.result is None:
            abort(404)
        r = job.result
        path = {
            "sheet": r.pdf or r.musicxml,
            "midi": r.drum_midi,
            "tab": r.bass_tab,
            "bmidi": r.bass_midi,
            "gtab": r.guitar_tab,
            "gmidi": r.guitar_midi,
            "gpb": r.bass_gp,
            "gpg": r.guitar_gp,
        }.get(kind)
        if path is None or not Path(path).exists():
            abort(404)
        path = Path(path)
        return send_file(path.resolve(), as_attachment=True, download_name=f"{job.title}-{kind}{path.suffix}")

    return app
