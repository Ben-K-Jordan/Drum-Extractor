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


def default_config_factory(output_dir: Path, model: str = "htdemucs_ft", device: str = "auto"):
    """Build the per-job pipeline config used by the web UI."""

    def make(job_id: str) -> PipelineConfig:
        return PipelineConfig(
            output_dir=output_dir / "jobs" / job_id,
            separation=SeparationConfig(model=model, device=device, stems=MIXER_STEMS),
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
            d["downloads"] = downloads
            d["warnings"] = self.result.warnings
        return d


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, title: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], title=title)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


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
            job.state = "done"
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
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # accept long WAV recordings
    store = JobStore()
    app.extensions["drum_extractor_jobs"] = store  # handy for tests

    @app.get("/")
    def index():
        return render_template("index.html", formats=", ".join(sorted(ALLOWED_EXTENSIONS)))

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
        # Server-controlled filename: keep only the sanitized stem + suffix.
        safe_stem = "".join(c for c in Path(file.filename).stem if c.isalnum() or c in "-_ ").strip() or "song"
        audio_path = upload_dir / f"{safe_stem}{suffix}"
        file.save(str(audio_path))

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
        }.get(kind)
        if path is None or not Path(path).exists():
            abort(404)
        path = Path(path)
        return send_file(path.resolve(), as_attachment=True, download_name=f"{job.title}-{kind}{path.suffix}")

    return app
