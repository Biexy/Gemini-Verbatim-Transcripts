#!/usr/bin/env bash
# One-command installer for Gemini Verbatim Transcripts (macOS & Linux).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Biexy/Gemini-Verbatim-Transcripts/main/install.sh | bash
#
# Optional environment overrides:
#   GVT_INSTALL_DIR  - Custom install directory (default: $HOME/.gemini-transcriber)
#   GVT_NO_LAUNCH    - Set to 1 to skip launching the Web UI after install

set -euo pipefail

REPO_OWNER="Biexy"
REPO_NAME="Gemini-Verbatim-Transcripts"
REPO_BRANCH="main"
REPO_GIT_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}.git"
REPO_TAR_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}/archive/refs/heads/${REPO_BRANCH}.tar.gz"

INSTALL_DIR="${GVT_INSTALL_DIR:-$HOME/.gemini-transcriber}"
BIN_DIR="$HOME/.local/bin"
VENV_DIR="$INSTALL_DIR/.venv"
NO_LAUNCH="${GVT_NO_LAUNCH:-0}"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
NC='\033[0m'

step() { printf "${CYAN}==> %s${NC}\n" "$1"; }
ok()   { printf "  ${GREEN}[OK]${NC} %s\n" "$1"; }
warn() { printf "  ${YELLOW}[WARN]${NC} %s\n" "$1"; }
err()  { printf "  ${RED}[ERROR]${NC} %s\n" "$1" >&2; }

find_python() {
    for cmd in python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            if "$cmd" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

ensure_ffmpeg() {
    step "Checking FFmpeg & FFprobe..."
    if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
        ok "FFmpeg ready ($(command -v ffmpeg))"
        return 0
    fi

    if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
        warn "FFmpeg not found. Installing via Homebrew..."
        brew install ffmpeg
    elif command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
        warn "FFmpeg not found. Installing via apt-get..."
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
    else
        warn "FFmpeg not found. Please install ffmpeg (e.g., brew install ffmpeg / sudo apt install ffmpeg)."
    fi
}

sync_repo() {
    step "Downloading ${REPO_NAME} into ${INSTALL_DIR}..."
    if command -v git >/dev/null 2>&1 && [[ -d "$INSTALL_DIR/.git" ]]; then
        git -C "$INSTALL_DIR" fetch origin "$REPO_BRANCH" --quiet
        git -C "$INSTALL_DIR" pull --ff-only origin "$REPO_BRANCH" --quiet
        ok "Updated existing installation via git"
        return 0
    elif command -v git >/dev/null 2>&1 && [[ ! -e "$INSTALL_DIR" ]]; then
        git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_GIT_URL" "$INSTALL_DIR" --quiet
        ok "Cloned repository via git"
        return 0
    fi

    local tmp_dir
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$tmp_dir"' EXIT
    curl -fsSL "$REPO_TAR_URL" -o "$tmp_dir/repo.tar.gz"
    tar -xzf "$tmp_dir/repo.tar.gz" -C "$tmp_dir"
    local inner
    inner="$(find "$tmp_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    mkdir -p "$INSTALL_DIR"
    cp -R "$inner"/. "$INSTALL_DIR"/
    ok "Installed source archive cleanly (no Git required)"
}

printf "\n${MAGENTA}==========================================================\n"
printf "   Gemini Verbatim Transcripts - One-Command Installer\n"
printf "==========================================================${NC}\n\n"

step "Checking Python 3.10+..."
if ! PY_BIN="$(find_python)"; then
    err "Python 3.10+ is required. Install Python 3.10+ and re-run."
    exit 1
fi
PY_VER="$("$PY_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
ok "Found Python ${PY_VER} (${PY_BIN})"

ensure_ffmpeg
sync_repo

step "Setting up isolated virtual environment (.venv)..."
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PY_BIN" -m venv "$VENV_DIR"
fi

step "Installing minimal dependencies (yt-dlp, google-genai, requests)..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip --quiet
"$VENV_DIR/bin/python" -m pip install -r "$INSTALL_DIR/requirements.txt" --quiet
"$VENV_DIR/bin/python" -m pip install --no-deps -e "$INSTALL_DIR" --quiet
ok "Environment & dependencies ready"

step "Installing CLI launcher into ${BIN_DIR}..."
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/gemini-transcribe" <<EOF
#!/usr/bin/env bash
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
exec "$VENV_DIR/bin/python" "$INSTALL_DIR/run.py" "\$@"
EOF
chmod +x "$BIN_DIR/gemini-transcribe"
ok "Installed command: ${BIN_DIR}/gemini-transcribe"

printf "\n${GREEN}==========================================================\n"
printf "  Installation Complete!\n"
printf "==========================================================${NC}\n"
printf "  Install folder : %s\n" "$INSTALL_DIR"
printf "  Run Web UI     : gemini-transcribe\n"
printf "  Run CLI        : gemini-transcribe --cli --course \"MyCourse\" --urls \"<URL>\"\n"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    printf "  ${YELLOW}Note:${NC} Add ~/.local/bin to your PATH: export PATH=\"\$HOME/.local/bin:\$PATH\"\n"
fi
printf "\n"

if [[ "$NO_LAUNCH" != "1" ]]; then
    step "Launching Gemini Verbatim Transcripts Web UI..."
    exec "$VENV_DIR/bin/python" "$INSTALL_DIR/run.py"
fi
