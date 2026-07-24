#!/usr/bin/env bash
# One-command installer for Drum Extractor (macOS / Linux).
#
#   ./install.sh            full install (separation + transcription + web UI)
#   ./install.sh --light    everything except the ML separation stack (no torch;
#                           useful for development or low-disk machines)
#
# Creates ./.venv, installs the right extras, tries the optional finicky pieces
# without failing the whole install, then runs `drum-extractor doctor`.

set -euo pipefail

# Everything below is relative to the repo root, so the script must work when
# invoked from anywhere (e.g. `bash /path/to/repo/install.sh`).
cd "$(dirname "$0")"

MODE="full"
VENV=".venv"
for arg in "$@"; do
  case "$arg" in
    --light) MODE="light" ;;
    --venv) shift_venv=1 ;;
    *) if [ "${shift_venv:-}" = 1 ]; then VENV="$arg"; shift_venv=""; else
         echo "unknown option: $arg (use --light, --venv DIR)"; exit 2; fi ;;
  esac
done
if [ "${shift_venv:-}" = 1 ]; then
  echo "--venv needs a directory argument"; exit 2
fi

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m  ! %s\033[0m\n' "$*"; }

# --- Python check -----------------------------------------------------------
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.10+ first (https://python.org)."; exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
  echo "Python 3.10+ required, found: $("$PY" --version)"; exit 1
fi
say "Using $("$PY" --version) at $(command -v "$PY")"

# --- venv -------------------------------------------------------------------
if [ ! -d "$VENV" ]; then
  say "Creating virtualenv at $VENV"
  "$PY" -m venv "$VENV"
fi
PIP="$VENV/bin/pip"
say "Upgrading pip"
"$PIP" install --quiet --upgrade pip

# --- main install -----------------------------------------------------------
if [ "$MODE" = "light" ]; then
  say "Installing (light: no ML separation stack — no torch/demucs/basic-pitch)"
  "$PIP" install -e ".[drums,notation,web,gp]"
else
  say "Installing (full — torch/demucs are large; the first download takes a while)"
  "$PIP" install -e ".[all]"
fi

# --- optional pieces that must not break the install ------------------------
say "Trying optional extras (safe to fail)"
if [ "$MODE" = "full" ]; then
  "$PIP" install -e ".[adtof]" \
    && echo "  ADTOF installed (best drum transcriber)" \
    || warn "ADTOF skipped (git install failed) — the onset fallback will be used; retry later with: $PIP install -e '.[adtof]'"
fi
"$PIP" install -e ".[quantize]" \
  && echo "  madmom installed (bar-accurate downbeats)" \
  || warn "madmom skipped (its 2018-era build often fails on modern Python) — librosa fallback is used automatically"

# --- pre-download model weights (full mode) ---------------------------------
# ~300 MB once; doing it now means the FIRST song doesn't stall on a silent
# download. Failure is fine — weights fetch automatically on first use.
if [ "$MODE" = "full" ]; then
  say "Pre-downloading the Demucs model (~300 MB, one time; safe to skip with Ctrl+C)"
  "$VENV/bin/python" - <<'PY' || warn "model download skipped — it will happen automatically on the first song"
from demucs.pretrained import get_model
get_model("htdemucs_ft")
print("model cached.")
PY
fi

# --- double-clickable launchers ----------------------------------------------
say "Creating launchers"
ROOT="$(pwd)"
# An absolute --venv path must not be glued onto ROOT.
case "$VENV" in
  /*) LAUNCH="$VENV" ;;
  *)  LAUNCH="$ROOT/$VENV" ;;
esac
cat > run.sh <<EOF
#!/usr/bin/env bash
exec "$LAUNCH/bin/drum-extractor" web "\$@"
EOF
chmod +x run.sh
echo "  ./run.sh"
if [ "$(uname -s)" = "Darwin" ]; then
  cp run.sh "Drum Extractor.command"
  chmod +x "Drum Extractor.command"
  echo "  'Drum Extractor.command'  (double-click it in Finder)"
fi

# --- verify -----------------------------------------------------------------
say "Checking the result"
"$VENV/bin/drum-extractor" doctor || true

cat <<EOF

Done. Start the app with:
  ./run.sh                            # browser opens automatically

Also useful:
  $VENV/bin/drum-extractor doctor     # re-check the environment any time

Optional (for PDF sheet export): install MuseScore 4 from https://musescore.org
EOF
