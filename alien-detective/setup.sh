#!/bin/bash
# ============================================================
#  Alien Signal Detective — One-Click Setup Script (macOS)
#  Run this once from the project root:  bash setup.sh
# ============================================================

set -e   # Exit on any error

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   🛸  Alien Signal Detective — Setup         ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. Check Python ───────────────────────────────────────────────────────────
echo -e "${YELLOW}[1/6] Checking Python version…${NC}"
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v $cmd &>/dev/null; then
        VER=$($cmd --version 2>&1)
        echo -e "  Found: $VER"
        PYTHON_CMD=$cmd
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}ERROR: Python 3.10+ not found.${NC}"
    echo "Install it with:  brew install python@3.12"
    exit 1
fi
echo -e "${GREEN}  ✓ Python OK${NC}"

# ── 2. Create virtual environment ─────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[2/6] Creating virtual environment…${NC}"
if [ -d "$VENV_DIR" ]; then
    echo "  .venv already exists — skipping creation"
else
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}  ✓ .venv created${NC}"
fi

source "$VENV_DIR/bin/activate"
echo -e "${GREEN}  ✓ Virtual environment activated${NC}"

# ── 3. Upgrade pip & install dependencies ────────────────────────────────────
echo ""
echo -e "${YELLOW}[3/6] Installing Python dependencies…${NC}"
pip install --upgrade pip --quiet
pip install -r "$PROJECT_DIR/requirements.txt" --quiet
echo -e "${GREEN}  ✓ All packages installed${NC}"

# ── 4. Generate dataset ───────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[4/6] Generating signal dataset…${NC}"
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/models"
cd "$PROJECT_DIR"
python -c "from src.data_generator import generate_dataset; generate_dataset(500, 'data/signals_dataset.csv')" 2>&1
echo -e "${GREEN}  ✓ Dataset generated → data/signals_dataset.csv${NC}"

# ── 5. Check Ollama ───────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[5/6] Checking Ollama (local LLM)…${NC}"
if command -v ollama &>/dev/null; then
    echo -e "${GREEN}  ✓ Ollama is installed${NC}"
    if ollama list 2>/dev/null | grep -q "llama3.2"; then
        echo -e "${GREEN}  ✓ llama3.2 model is available${NC}"
    else
        echo -e "${YELLOW}  → llama3.2 not found. Pulling now (this may take a few minutes)…${NC}"
        ollama pull llama3.2 || echo -e "${YELLOW}  ⚠ Pull failed — start Ollama app first, then run: ollama pull llama3.2${NC}"
    fi
else
    echo -e "${YELLOW}  ⚠ Ollama not installed.${NC}"
    echo "  Install from: https://ollama.com/download"
    echo "  Then run:     ollama pull llama3.2"
    echo "  The game works without ARIA, but NLP features will show offline messages."
fi

# ── 6. Done ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[6/6] Setup complete!${NC}"
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   To launch the game:                        ║${NC}"
echo -e "${CYAN}║                                              ║${NC}"
echo -e "${CYAN}║   source .venv/bin/activate                  ║${NC}"
echo -e "${CYAN}║   streamlit run app.py                       ║${NC}"
echo -e "${CYAN}║                                              ║${NC}"
echo -e "${CYAN}║   Or in VS Code: press F5                    ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# Offer to launch immediately
read -p "Launch the game now? (y/n): " LAUNCH
if [[ "$LAUNCH" == "y" || "$LAUNCH" == "Y" ]]; then
    echo -e "${GREEN}Starting Streamlit…${NC}"
    cd "$PROJECT_DIR"
    streamlit run app.py
fi
