#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# ANIMA Installer
# ─────────────────────────────────────────────────────────────

ANIMA_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ANIMA_ROOT/persistence-env"
CONFIG_DIR="$ANIMA_ROOT/config"
DATA_DIR="$ANIMA_ROOT/data"

# ─── Release Config ───
GITHUB_REPO="SomeNewGuy/anima-public"
CORE_VERSION="0.1.3"

# ─── Detect Platform ───
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
    Linux)
        case "$ARCH" in
            x86_64)  CORE_WHEEL="anima_core-${CORE_VERSION}-cp312-cp312-manylinux_2_34_x86_64.whl" ;;
            aarch64) CORE_WHEEL="anima_core-${CORE_VERSION}-cp312-cp312-manylinux_2_34_aarch64.whl" ;;
            *)       echo "  ERROR: Unsupported Linux architecture: $ARCH"; exit 1 ;;
        esac
        ;;
    Darwin)
        case "$ARCH" in
            x86_64)  CORE_WHEEL="anima_core-${CORE_VERSION}-cp312-cp312-macosx_10_12_x86_64.whl" ;;
            arm64)   CORE_WHEEL="anima_core-${CORE_VERSION}-cp312-cp312-macosx_11_0_arm64.whl" ;;
            *)       echo "  ERROR: Unsupported macOS architecture: $ARCH"; exit 1 ;;
        esac
        ;;
    MINGW*|MSYS*|CYGWIN*)
        CORE_WHEEL="anima_core-${CORE_VERSION}-cp312-cp312-win_amd64.whl"
        ;;
    *)
        echo "  ERROR: Unsupported OS: $OS"
        echo "  Supported: Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (amd64)"
        exit 1
        ;;
esac
echo "  Platform: $OS/$ARCH → $CORE_WHEEL"

echo ""
echo "  ╔═══════════════════════════════════════╗"
echo "  ║          ANIMA Platform Setup         ║"
echo "  ╚═══════════════════════════════════════╝"
echo ""

# ─── Check Python ───
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "  ERROR: Python 3.10+ required. Found none."
    echo "  Install Python 3.10+ and try again."
    exit 1
fi
echo "  [1/8] Python: $($PYTHON --version)"

# ─── Create Virtual Environment ───
if [ -d "$VENV_DIR" ]; then
    echo "  [2/8] Virtual environment exists: $VENV_DIR"
else
    echo "  [2/8] Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ─── Install Dependencies ───
echo "  [3/8] Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip -q
pip install -r "$ANIMA_ROOT/requirements.txt" -q

# ─── Install ANIMA Core (Rust binary) ───
echo "  [4/8] Installing ANIMA Core..."
CORE_URL="https://github.com/${GITHUB_REPO}/releases/download/v${CORE_VERSION}/${CORE_WHEEL}"
if python -c "import anima_core" 2>/dev/null; then
    echo "         anima_core already installed"
else
    WHEEL_PATH="$ANIMA_ROOT/$CORE_WHEEL"
    if curl -fSL "$CORE_URL" -o "$WHEEL_PATH" 2>/dev/null; then
        pip install "$WHEEL_PATH" -q
        rm -f "$WHEEL_PATH"
        echo "         anima_core installed"
    else
        echo "  ERROR: Could not download anima_core for $OS/$ARCH."
        echo "         Your platform may not have a binary available yet."
        echo "         Available: Linux x86_64. macOS and Windows coming soon."
        echo "         Check: https://github.com/${GITHUB_REPO}/releases"
        exit 1
    fi
fi

# ─── Download spaCy Model ───
echo "  [5/8] Downloading spaCy language model..."
python -m spacy download en_core_web_sm -q 2>/dev/null || echo "         (spaCy model download failed — install manually: python -m spacy download en_core_web_sm)"

# ─── Create Config ───
if [ -f "$CONFIG_DIR/settings.toml" ]; then
    echo "  [6/8] Config exists: $CONFIG_DIR/settings.toml"
else
    echo "  [6/8] Creating config from template..."
    cp "$CONFIG_DIR/settings.example.toml" "$CONFIG_DIR/settings.toml"
    echo "         Edit config/settings.toml to add your inference server."
fi

# ─── Create .env ───
if [ -f "$ANIMA_ROOT/.env" ]; then
    echo "  [6b]  .env exists"
else
    cp "$ANIMA_ROOT/.env.example" "$ANIMA_ROOT/.env"
fi

# ─── Create Data Directories ───
echo "  [7/8] Creating data directories..."
mkdir -p "$DATA_DIR/sqlite"
mkdir -p "$DATA_DIR/chroma"
mkdir -p "$DATA_DIR/logs"

# ─── Sanity Check ───
echo "  [8/8] Running sanity check..."
IMPORT_OK=true
python -c "
import fastapi, uvicorn, pydantic, requests, toml, rich
print('         Core imports: OK')
" 2>/dev/null || { echo "         Core imports: FAILED"; IMPORT_OK=false; }

python -c "
import chromadb, sentence_transformers
print('         Memory imports: OK')
" 2>/dev/null || { echo "         Memory imports: FAILED (embedding search may not work)"; }

python -c "
import anima_core
print('         Rust core: OK')
" 2>/dev/null || { echo "         Rust core: FAILED — anima_core is required"; IMPORT_OK=false; }

echo ""
if [ "$IMPORT_OK" = true ]; then
    echo "  ────────────────────────────────────────"
    echo "  ANIMA installed successfully."
    echo ""
    echo "  Next steps:"
    echo "    1. Edit config/settings.toml"
    echo "       - Set your inference server endpoint"
    echo "       - See docs/MODELS.md for setup help"
    echo ""
    echo "    2. Start ANIMA:"
    echo "       ./anima start"
    echo ""
    echo "    3. Open dashboard:"
    echo "       http://localhost:8900"
    echo "  ────────────────────────────────────────"
else
    echo "  ────────────────────────────────────────"
    echo "  Installation completed with errors."
    echo "  Check the output above and fix any issues."
    echo "  ────────────────────────────────────────"
    exit 1
fi
