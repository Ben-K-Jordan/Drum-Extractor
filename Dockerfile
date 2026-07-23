# Drum Extractor — containerized web UI (CPU).
#
#   docker build -t drum-extractor .
#   docker run -p 8237:8237 -v drumx-output:/app/output drum-extractor
#
# Then open http://127.0.0.1:8237. Separation runs on CPU in the container
# (roughly the track's length per song); for GPU speed, install natively with
# ./install.sh instead.

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY drum_extractor ./drum_extractor

RUN pip install --no-cache-dir -e ".[all]" \
    && (pip install --no-cache-dir -e ".[adtof]" || echo "ADTOF skipped; onset fallback will be used")

EXPOSE 8237
VOLUME ["/app/output"]

CMD ["drum-extractor", "web", "--host", "0.0.0.0", "--port", "8237", "--no-browser"]
