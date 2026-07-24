"""End-to-end integration test using only the dependency-light fallback backends.

Generates a synthetic drum stem, then runs the real onset transcriber, librosa
quantizer, and music21 notation stage. Skips cleanly if the audio/notation deps
aren't installed. This is the test that actually exercises the pipeline wiring
(not just the pure-Python units).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("librosa")
pytest.importorskip("soundfile")
pytest.importorskip("pretty_midi")
pytest.importorskip("music21")

import soundfile as sf  # noqa: E402

from drum_extractor.config import (  # noqa: E402
    BassTranscriptionConfig,
    DrumTranscriptionConfig,
    NotationConfig,
    PipelineConfig,
    QuantizeConfig,
)
from drum_extractor.pipeline import run_pipeline  # noqa: E402


def _synth_drum_stem(path, bpm=120.0, bars=2, sr=44100):
    spb = 60.0 / bpm
    n = int((bars * 4 * spb + 0.5) * sr)
    audio = np.zeros(n)
    rng = np.random.default_rng(0)

    def add(t, snd):
        i = int(t * sr)
        j = min(i + len(snd), n)
        audio[i:j] += snd[: j - i]

    def kick():
        t = np.linspace(0, 0.15, int(0.15 * sr))
        freq = 120 * np.exp(-t * 20) + 45
        return 0.9 * np.exp(-t * 30) * np.sin(2 * np.pi * np.cumsum(freq) / sr)

    def snare():
        t = np.linspace(0, 0.12, int(0.12 * sr))
        return 0.6 * np.exp(-t * 25) * (0.7 * rng.standard_normal(len(t)) + 0.3 * np.sin(2 * np.pi * 190 * t))

    def hihat():
        t = np.linspace(0, 0.05, int(0.05 * sr))
        return 0.25 * np.exp(-t * 80) * np.diff(rng.standard_normal(len(t)), prepend=0)

    for bar in range(bars):
        base = bar * 4 * spb
        for i in range(8):
            add(base + i * spb / 2, hihat())
        add(base + 0 * spb, kick())
        add(base + 2 * spb, kick())
        add(base + 1 * spb, snare())
        add(base + 3 * spb, snare())
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.9
    sf.write(str(path), audio.astype(np.float32), sr)


def test_fallback_pipeline_end_to_end(tmp_path):
    # Place a synthetic drum stem where the pipeline expects it (separation off).
    song_dir = tmp_path / "synth"
    (song_dir / "stems").mkdir(parents=True)
    _synth_drum_stem(song_dir / "stems" / "drums.wav")

    config = PipelineConfig(
        output_dir=tmp_path,
        do_separation=False,
        drums=DrumTranscriptionConfig(backend="onset"),
        bass=BassTranscriptionConfig(backend="none"),
        quantize=QuantizeConfig(backend="librosa", grid=16),
        notation=NotationConfig(render_pdf=False),
    )
    result = run_pipeline(song_dir / "synth.wav", config)

    # Transcription produced hits and detected a plausible tempo.
    assert len(result.transcription.drum_hits) > 0
    assert result.transcription.tempo is not None
    assert 100 < result.transcription.tempo < 140  # synth is 120 BPM

    # No duplicate (instrument, time) hits survived quantization.
    keys = [(h.instrument, round(h.time, 4)) for h in result.transcription.drum_hits]
    assert len(keys) == len(set(keys))

    # Every hit was assigned a bar and beat.
    assert all(h.bar is not None and h.beat is not None for h in result.transcription.drum_hits)

    # Artifacts exist and the MusicXML is a real score.
    assert result.drum_midi and result.drum_midi.exists()
    assert result.musicxml and result.musicxml.exists()
    assert "score-partwise" in result.musicxml.read_text()


def test_pipeline_with_phase4_booster_and_sonify(tmp_path):
    """Full fallback pipeline with the double-kick booster and correction aids on."""
    song_dir = tmp_path / "metal"
    (song_dir / "stems").mkdir(parents=True)
    _synth_drum_stem(song_dir / "stems" / "drums.wav")

    config = PipelineConfig(
        output_dir=tmp_path,
        do_separation=False,
        drums=DrumTranscriptionConfig(backend="onset", boost_double_kick=True),
        bass=BassTranscriptionConfig(backend="none"),
        quantize=QuantizeConfig(backend="librosa", grid=32),
        notation=NotationConfig(render_pdf=False),
        do_sonify=True,
    )
    result = run_pipeline(song_dir / "metal.wav", config)

    assert len(result.transcription.drum_hits) > 0
    # Correction aids were produced and are real files.
    assert result.drum_sonification and result.drum_sonification.exists()
    assert result.onset_csv and result.onset_csv.exists()
    assert result.onset_csv.read_text().startswith("time_sec,instrument,velocity_midi")
