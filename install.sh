#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# ANIMA Installer
# ─────────────────────────────────────────────────────────────

ANIMA_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ANIMA_ROOT/persistence-env"
CONFIG_DIR="$ANIMA_ROOT/config"
DATA_DIR="$ANIMA_ROOT/data"

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
echo "  [1/7] Python: $($PYTHON --version)"

# ─── Create Virtual Environment ───
if [ -d "$VENV_DIR" ]; then
    echo "  [2/7] Virtual environment exists: $VENV_DIR"
else
    echo "  [2/7] Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ─── Install Dependencies ───
echo "  [3/7] Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip -q
pip install -r "$ANIMA_ROOT/requirements.txt" -q

# ─── Download spaCy Model ───
echo "  [4/7] Downloading spaCy language model..."
python -m spacy download en_core_web_sm -q 2>/dev/null || echo "         (spaCy model download failed — install manually: python -m spacy download en_core_web_sm)"

# ─── Create Config ───
if [ -f "$CONFIG_DIR/settings.toml" ]; then
    echo "  [5/7] Config exists: $CONFIG_DIR/settings.toml"
else
    echo "  [5/7] Creating config from template..."
    cp "$CONFIG_DIR/settings.example.toml" "$CONFIG_DIR/settings.toml"
    echo "         Edit config/settings.toml to add your inference server."
fi

# ─── Create .env ───
if [ -f "$ANIMA_ROOT/.env" ]; then
    echo "  [5b]  .env exists"
else
    cp "$ANIMA_ROOT/.env.example" "$ANIMA_ROOT/.env"
fi

# ─── Create Data Directories ───
echo "  [6/7] Creating data directories..."
mkdir -p "$DATA_DIR/sqlite"
mkdir -p "$DATA_DIR/chroma"
mkdir -p "$DATA_DIR/logs"

# ─── Sanity Check ───
echo "  [7/7] Running sanity check..."
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
try:
    import anima_core
    print('         Rust core: LOADED (dreams enabled)')
except ImportError:
    print('         Rust core: not installed (memory-only mode)')
    print('         Download from: https://animahub.io/core')
" 2>/dev/null

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
