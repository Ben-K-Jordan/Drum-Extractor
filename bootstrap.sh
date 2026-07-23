#!/usr/bin/env bash
# Drum Extractor one-liner bootstrap:
#
#   curl -fsSL https://raw.githubusercontent.com/Ben-K-Jordan/Drum-Extractor/main/bootstrap.sh | bash
#
# Clones (or updates) the repo into ~/drum-extractor, runs the installer, and
# launches the web app — the browser opens automatically.
#
# Overridable via environment: DRUMX_DIR, DRUMX_REPO, DRUMX_INSTALL_ARGS
# (e.g. "--light"), DRUMX_NO_LAUNCH=1 to install without starting the app.

set -euo pipefail

REPO="${DRUMX_REPO:-https://github.com/Ben-K-Jordan/Drum-Extractor.git}"
DIR="${DRUMX_DIR:-$HOME/drum-extractor}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

for tool in git curl; do
  if ! command -v "$tool" >/dev/null 2>&1 && [ "$tool" = git ]; then
    echo "git is required. Install it (macOS: xcode-select --install; Debian/Ubuntu: sudo apt install git) and re-run."
    exit 1
  fi
done

if [ -d "$DIR/.git" ]; then
  say "Updating existing install at $DIR"
  git -C "$DIR" pull --ff-only || echo "  (couldn't fast-forward; keeping your current version)"
else
  say "Cloning into $DIR"
  git clone --depth 1 "$REPO" "$DIR"
fi

cd "$DIR"
# shellcheck disable=SC2086
./install.sh ${DRUMX_INSTALL_ARGS:-}

if [ "${DRUMX_NO_LAUNCH:-0}" = 1 ]; then
  say "Installed. Start any time with: $DIR/run.sh"
else
  say "Launching Drum Extractor (Ctrl+C to stop; restart later with $DIR/run.sh)"
  exec "$DIR/.venv/bin/drum-extractor" web
fi
