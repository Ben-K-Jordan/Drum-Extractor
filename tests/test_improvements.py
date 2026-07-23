"""Tests for the pre-real-song sprint: progress, velocity/ghosts, persistence,
timing overlays, and Guitar Pro export."""

from __future__ import annotations

from pathlib import Path

import pytest

from drum_extractor.config import BassTranscriptionConfig, NotationConfig, QuantizeConfig
from drum_extractor.events import BassNote, DrumHit, Transcription


# --- separation progress ---------------------------------------------------------------

def test_demucs_progress_fraction_math():
    from drum_extractor.separation import _demucs_progress_fraction

    # Mid-way through the second of four bag models.
    f = _demucs_progress_fraction({"models": 4, "model_idx_in_bag": 1, "segment_offset": 50.0, "audio_length": 100.0})
    assert abs(f - (1 + 0.5) / 4) < 1e-9
    assert _demucs_progress_fraction({}) == 0.0 or _demucs_progress_fraction({}) is not None
    assert _demucs_progress_fraction({"models": 1, "segment_offset": 999, "audio_length": 10}) <= 1.0


# --- velocity / ghost notes --------------------------------------------------------------

def test_onset_fallback_scales_velocity_with_energy(tmp_path):
    pytest.importorskip("librosa")
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf
    from drum_extractor.config import DrumTranscriptionConfig
    from drum_extractor.drums import _transcribe_onsets

    sr = 44100
    rng = np.random.default_rng(3)
    audio = np.zeros(3 * sr, dtype="float32")
    t = np.linspace(0, 0.12, int(0.12 * sr))
    # A percussive kick: low fundamental + a short broadband click so the onset
    # detector fires cleanly on every hit (pure low sines have too soft an attack).
    body = np.exp(-t * 30) * np.sin(2 * np.pi * 60 * t)
    click = np.zeros_like(body)
    click[:300] = rng.standard_normal(300) * 0.15
    kick = body + click
    onsets = [0.3 + k * 0.4 for k in range(6)]
    amps = [1.0, 0.15, 1.0, 0.15, 1.0, 0.15]  # accent / ghost alternation
    for t0, amp in zip(onsets, amps):
        i = int(t0 * sr)
        audio[i : i + len(kick)] += (amp * kick).astype("float32")
    p = tmp_path / "kicks.wav"
    sf.write(str(p), audio, sr)

    hits = _transcribe_onsets(p, DrumTranscriptionConfig())
    # Velocity is what's under test (band classification of synthetic clicks is
    # not): take the max velocity near each planted onset.
    vel_at = []
    for t0 in onsets:
        near = [h.velocity for h in hits if abs(h.time - t0) < 0.15]
        assert near, f"no hit detected near {t0}"
        vel_at.append(max(near))
    louds, softs = vel_at[0::2], vel_at[1::2]
    assert min(louds) > max(softs) + 15, f"velocities not separated: {louds} vs {softs}"


def test_notation_marks_ghosts_and_accents(tmp_path):
    pytest.importorskip("music21")
    from drum_extractor.notation import transcription_to_musicxml

    hits = [
        DrumHit(0.0, "snare", velocity=30, bar=1, beat=0.0),   # ghost
        DrumHit(0.0, "snare", velocity=120, bar=1, beat=2.0),  # accent
    ]
    tr = Transcription(drum_hits=hits, tempo=120.0)
    out = transcription_to_musicxml(tr, tmp_path / "d.musicxml", NotationConfig(), QuantizeConfig())
    xml = out.read_text()
    assert 'parentheses="yes"' in xml  # ghost notehead
    assert "<accent" in xml  # accent articulation


# --- Guitar Pro export -------------------------------------------------------------------

def test_gp5_round_trip(tmp_path):
    pytest.importorskip("guitarpro")
    import guitarpro as gp
    from drum_extractor.gp_export import write_gp5
    from drum_extractor.tabs import assign_frets

    cfg = BassTranscriptionConfig()
    notes = [BassNote(0.0, 0.4, 33), BassNote(0.5, 0.9, 40), BassNote(1.0, 1.4, 45)]
    assign_frets(notes, cfg.tuning, cfg.frets)
    out = write_gp5(notes, cfg.tuning, tmp_path / "b.gp5", tempo=120.0, track_name="Bass")

    song = gp.parse(str(out))
    back = []
    for m in song.tracks[0].measures:
        for beat in m.voices[0].beats:
            for n in beat.notes:
                # gp string numbers count from the top; convert back to ours.
                back.append(song.tracks[0].strings[n.string - 1].value + n.value)
    assert sorted(back) == [33, 40, 45]  # same pitches survive the format


def test_gp5_no_fretted_notes_raises(tmp_path):
    pytest.importorskip("guitarpro")
    from drum_extractor.gp_export import write_gp5

    with pytest.raises(ValueError):
        write_gp5([BassNote(0.0, 0.5, 5)], (28, 33, 38, 43), tmp_path / "x.gp5")  # unplaceable only


# --- web: persistence + timing -------------------------------------------------------------

@pytest.fixture()
def web_bits(tmp_path, monkeypatch):
    pytest.importorskip("flask")
    from drum_extractor.config import PipelineConfig
    from drum_extractor.pipeline import PipelineResult
    from drum_extractor.web import server as web_server

    def fake_pipeline(audio_path, config, on_stage=None):
        stems_dir = Path(config.output_dir) / "stems"
        stems_dir.mkdir(parents=True, exist_ok=True)
        result = PipelineResult()
        for name in ("drums", "bass"):
            p = stems_dir / f"{name}.wav"
            p.write_bytes(b"RIFFxxxx")
            setattr(result.stems, name, p)
        result.transcription = Transcription(
            drum_hits=[DrumHit(0.0, "kick", 100), DrumHit(0.5, "snare", 40)],
            tempo=120.0, downbeats=[0.0, 2.0],
        )
        return result

    monkeypatch.setattr(web_server, "run_pipeline", fake_pipeline)
    factory = lambda job_id, guitar=False: PipelineConfig(output_dir=tmp_path / "jobs" / job_id)  # noqa: E731
    return web_server, factory, tmp_path


def _upload(client):
    import io

    return client.post("/upload", data={"song": (io.BytesIO(b"RIFF"), "song.wav")},
                       content_type="multipart/form-data")


def test_timing_exposed_to_mixer(web_bits):
    web_server, factory, tmp_path = web_bits
    app = web_server.create_app(config_factory=factory, output_dir=tmp_path, sync=True)
    client = app.test_client()
    job_id = _upload(client).get_json()["id"]
    job = client.get(f"/job/{job_id}").get_json()
    assert job["timing"]["tempo"] == 120.0
    assert job["timing"]["downbeats"] == [0.0, 2.0]
    assert ["0.0", "kick"] != job["timing"]["drum_hits"][0]  # times are numbers
    assert job["timing"]["drum_hits"][0] == [0.0, "kick"]


def test_jobs_survive_server_restart(web_bits):
    web_server, factory, tmp_path = web_bits
    app1 = web_server.create_app(config_factory=factory, output_dir=tmp_path, sync=True)
    client1 = app1.test_client()
    job_id = _upload(client1).get_json()["id"]
    assert client1.get(f"/job/{job_id}").get_json()["state"] == "done"

    # A brand-new app instance (server restart) must still know the song.
    app2 = web_server.create_app(config_factory=factory, output_dir=tmp_path, sync=True)
    client2 = app2.test_client()
    job = client2.get(f"/job/{job_id}").get_json()
    assert job["state"] == "done"
    assert set(job["stems"]) == {"drums", "bass"}
    assert job["timing"]["tempo"] == 120.0
    assert client2.get(f"/stems/{job_id}/drums").status_code == 200
    assert client2.get(f"/mixer/{job_id}").status_code == 200
    # And it shows up in the library on the drop page.
    assert b"song" in client2.get("/").data


def test_restore_skips_deleted_stems(web_bits):
    import shutil

    web_server, factory, tmp_path = web_bits
    app1 = web_server.create_app(config_factory=factory, output_dir=tmp_path, sync=True)
    job_id = _upload(app1.test_client()).get_json()["id"]
    shutil.rmtree(tmp_path / "jobs" / job_id / "stems")

    app2 = web_server.create_app(config_factory=factory, output_dir=tmp_path, sync=True)
    assert app2.test_client().get(f"/job/{job_id}").status_code == 404
