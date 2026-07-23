"""Tests for the local web UI (upload -> job -> mixer -> stems -> downloads).

Uses ``sync=True`` (jobs run inline) and a stubbed pipeline, so the flow is
deterministic and needs no audio models.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flask")

from drum_extractor.config import PipelineConfig  # noqa: E402
from drum_extractor.events import Stems  # noqa: E402
from drum_extractor.pipeline import PipelineResult  # noqa: E402
from drum_extractor.web import server as web_server  # noqa: E402


@pytest.fixture()
def app(tmp_path, monkeypatch):
    def fake_pipeline(audio_path, config, on_stage=None):
        if on_stage:
            on_stage("separating")
        stems_dir = Path(config.output_dir) / "stems"
        stems_dir.mkdir(parents=True, exist_ok=True)
        result = PipelineResult()
        for name in ("drums", "bass"):
            p = stems_dir / f"{name}.wav"
            p.write_bytes(b"RIFFfake-wav-payload")
            setattr(result.stems, name, p)
        mx = Path(config.output_dir) / "drums.musicxml"
        mx.write_text("<score-partwise/>")
        result.musicxml = mx
        mid = Path(config.output_dir) / "drums.mid"
        mid.write_bytes(b"MThd fake")
        result.drum_midi = mid
        return result

    monkeypatch.setattr(web_server, "run_pipeline", fake_pipeline)
    factory = lambda job_id: PipelineConfig(output_dir=tmp_path / "jobs" / job_id)  # noqa: E731
    flask_app = web_server.create_app(config_factory=factory, output_dir=tmp_path, sync=True)
    flask_app.config["TESTING"] = True
    return flask_app


def _upload(client, filename="song.wav"):
    return client.post(
        "/upload",
        data={"song": (__import__("io").BytesIO(b"RIFF-fake-audio"), filename)},
        content_type="multipart/form-data",
    )


def test_index_page(app):
    res = app.test_client().get("/")
    assert res.status_code == 200
    assert b"Drop a song" in res.data


def test_upload_rejects_unsupported_type(app):
    res = _upload(app.test_client(), "notes.txt")
    assert res.status_code == 400
    assert "Unsupported" in res.get_json()["error"]


def test_upload_rejects_missing_file(app):
    res = app.test_client().post("/upload", data={}, content_type="multipart/form-data")
    assert res.status_code == 400


def test_full_flow(app):
    client = app.test_client()
    res = _upload(client)
    assert res.status_code == 200
    data = res.get_json()
    job_id = data["id"]
    assert data["mixer"] == f"/mixer/{job_id}"

    # Sync mode: the stubbed job is already done.
    job = client.get(f"/job/{job_id}").get_json()
    assert job["state"] == "done"
    assert set(job["stems"]) == {"drums", "bass"}
    assert "sheet" in job["downloads"] and "midi" in job["downloads"]
    assert "MusicXML" in job["downloads"]["sheet"]["label"]  # no PDF in the stub

    # Mixer page renders.
    page = client.get(f"/mixer/{job_id}")
    assert page.status_code == 200
    assert b"Download mix" in page.data

    # Stems served as audio.
    stem = client.get(f"/stems/{job_id}/drums")
    assert stem.status_code == 200
    assert stem.mimetype == "audio/wav"
    assert stem.data.startswith(b"RIFF")

    # Downloads work and are attachments.
    sheet = client.get(f"/download/{job_id}/sheet")
    assert sheet.status_code == 200
    assert "attachment" in sheet.headers.get("Content-Disposition", "")
    assert client.get(f"/download/{job_id}/midi").status_code == 200


def test_unknown_ids_and_names_404(app):
    client = app.test_client()
    assert client.get("/job/nope").status_code == 404
    assert client.get("/mixer/nope").status_code == 404

    job_id = _upload(client).get_json()["id"]
    # Stem names resolve only through the job's own results — no path tricks.
    assert client.get(f"/stems/{job_id}/vocals").status_code == 404  # stub made no vocals
    assert client.get(f"/stems/{job_id}/..%2f..%2fsecrets").status_code == 404
    assert client.get(f"/download/{job_id}/tab").status_code == 404  # stub made no tab
    assert client.get(f"/download/{job_id}/passwd").status_code == 404


def test_pipeline_failure_reported(app, monkeypatch):
    def boom(audio_path, config, on_stage=None):
        raise RuntimeError("demucs exploded")

    monkeypatch.setattr(web_server, "run_pipeline", boom)
    client = app.test_client()
    job_id = _upload(client).get_json()["id"]
    job = client.get(f"/job/{job_id}").get_json()
    assert job["state"] == "error"
    assert "demucs exploded" in job["error"]


def test_cli_web_flags_parse():
    from drum_extractor.cli import build_parser

    args = build_parser().parse_args(["web", "--port", "9000", "--model", "htdemucs"])
    assert args.command == "web"
    assert args.port == 9000
