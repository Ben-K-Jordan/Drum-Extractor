# Drum Extractor — containerized web UI (CPU).
#
#   docker build -t drum-extractor .
#   docker run -p 8237:8237 -v drumx-output:/app/output drum-extractor
#
# Then open http://127.0.0.1:8237. Separation runs on CPU in the container
# (roughly the track's length per song); for GPU speed, install natively with
# ./install.sh instead.
#
# Image-size notes (learned the hard way — the first build ran for an hour):
# - torch MUST come from the CPU wheel index: the default PyPI wheel bundles
#   ~2 GB of CUDA libraries a container never uses.
# - the `ensemble` extra (audio-separator + onnxruntime) is deliberately
#   excluded; it's an optional two-model blend, not worth the weight here.

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY drum_extractor ./drum_extractor

RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e ".[separation,drums,bass,bass-crepe,notation,web,gp]" \
    && (pip install --no-cache-dir -e ".[adtof]" || echo "ADTOF skipped; onset fallback will be used") \
    && rm -rf /root/.cache

# Bake the Demucs weights in (~320 MB): without this the FIRST song uploaded
# to a fresh container stalls for minutes on a silent model download. Worth
# the image size for a tool whose pitch is "docker run and drop a song in".
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs_ft')" \
    || echo "weights pre-download failed; they will download on first use"

EXPOSE 8237
VOLUME ["/app/output"]

CMD ["drum-extractor", "web", "--host", "0.0.0.0", "--port", "8237", "--no-browser"]
