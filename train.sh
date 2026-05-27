#!/bin/bash
# ============================================================
# train.sh — One-click Bi-LSTM training for cancer-pipeline
#
# What this script does:
#   1. Checks the dataset exists (or downloads it via Kaggle API)
#   2. Trains the Bidirectional LSTM model
#   3. Saves bilstm_cancer_classifier.h5 → models/
#   4. Saves tokenizer.pkl              → models/
#   5. Saves training_history.png       → models/
#
# Usage:
#   chmod +x train.sh
#   ./train.sh
#
# Optional — custom epochs and batch size:
#   ./train.sh --epochs 15 --batch-size 64
# ============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "${GREEN}✅  $1${NC}"; }
info() { echo -e "${YELLOW}➜  $1${NC}"; }
fail() { echo -e "${RED}❌  $1${NC}"; exit 1; }

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   cancer-pipeline  —  Bi-LSTM Training      ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"
echo ""

# Move to project root
cd "$(dirname "$0")"

# ════════════════════════════════════════════════════════════
# STEP A — Check dataset exists
# ════════════════════════════════════════════════════════════
info "Checking for dataset..."

CSV_PATH="data/raw/clinical_text.csv"

if [ -f "$CSV_PATH" ]; then
    ROW_COUNT=$(wc -l < "$CSV_PATH")
    ok "Dataset found: $CSV_PATH ($ROW_COUNT lines)"
else
    info "Dataset not found. Attempting Kaggle API download..."

    # Load .env to get Kaggle credentials
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | grep -v '^$' | xargs)
    fi

    if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
        echo ""
        echo -e "${RED}╔══════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║  Kaggle credentials not found in .env                    ║${NC}"
        echo -e "${RED}╠══════════════════════════════════════════════════════════╣${NC}"
        echo -e "${RED}║  You have 2 options:                                     ║${NC}"
        echo -e "${RED}║                                                          ║${NC}"
        echo -e "${RED}║  Option A — Add Kaggle credentials to .env:              ║${NC}"
        echo -e "${RED}║    1. Go to https://www.kaggle.com/settings              ║${NC}"
        echo -e "${RED}║    2. Click API → Create New Token → downloads json      ║${NC}"
        echo -e "${RED}║    3. Open .env and fill in:                             ║${NC}"
        echo -e "${RED}║       KAGGLE_USERNAME=your_username                      ║${NC}"
        echo -e "${RED}║       KAGGLE_KEY=your_api_key                            ║${NC}"
        echo -e "${RED}║    4. Re-run: ./train.sh                                 ║${NC}"
        echo -e "${RED}║                                                          ║${NC}"
        echo -e "${RED}║  Option B — Download manually from Kaggle:               ║${NC}"
        echo -e "${RED}║    1. Visit this URL:                                    ║${NC}"
        echo -e "${RED}║       https://kaggle.com/datasets/ritheshsreenivasan/    ║${NC}"
        echo -e "${RED}║       clinical-text-classification                       ║${NC}"
        echo -e "${RED}║    2. Download and unzip                                 ║${NC}"
        echo -e "${RED}║    3. Place the CSV at: data/raw/clinical_text.csv       ║${NC}"
        echo -e "${RED}║    4. Re-run: ./train.sh                                 ║${NC}"
        echo -e "${RED}╚══════════════════════════════════════════════════════════╝${NC}"
        echo ""
        exit 1
    fi

    # Download via Kaggle API
    info "Downloading dataset from Kaggle..."
    python3 -c "
from src.ingestion.kaggle_downloader import download_dataset
path = download_dataset()
print(f'Downloaded to: {path}')
"
    ok "Dataset downloaded successfully."
fi

# ════════════════════════════════════════════════════════════
# STEP B — Check models/ directory exists
# ════════════════════════════════════════════════════════════
mkdir -p models
ok "models/ directory ready."

# ════════════════════════════════════════════════════════════
# STEP C — Run training
# ════════════════════════════════════════════════════════════
info "Starting Bi-LSTM training..."
echo ""
echo "  Architecture: Embedding → BiLSTM(128) → BiLSTM(64) → Dense(3, softmax)"
echo "  Dataset:      7,500+ clinical text records"
echo "  Classes:      Thyroid / Colon / Lung Cancer"
echo ""
echo "  ⏱  Estimated time: 5–15 minutes depending on your Mac"
echo ""

# Pass any extra CLI args (--epochs, --batch-size) to the Python script
python3 src/model/train.py "$@"

# ════════════════════════════════════════════════════════════
# STEP D — Verify output files were created
# ════════════════════════════════════════════════════════════
echo ""
info "Verifying output files..."

ALL_OK=true

if [ -f "models/bilstm_cancer_classifier.h5" ]; then
    SIZE=$(du -sh models/bilstm_cancer_classifier.h5 | cut -f1)
    ok "models/bilstm_cancer_classifier.h5  ($SIZE)"
else
    echo -e "${RED}❌  models/bilstm_cancer_classifier.h5 NOT found${NC}"
    ALL_OK=false
fi

if [ -f "models/tokenizer.pkl" ]; then
    SIZE=$(du -sh models/tokenizer.pkl | cut -f1)
    ok "models/tokenizer.pkl  ($SIZE)"
else
    echo -e "${RED}❌  models/tokenizer.pkl NOT found${NC}"
    ALL_OK=false
fi

if [ -f "models/training_history.png" ]; then
    ok "models/training_history.png  (accuracy & loss curves)"
fi

# ════════════════════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════════════════════
echo ""
if $ALL_OK; then
    echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${GREEN}║         Step 2 Complete! 🎉                  ║${NC}"
    echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}Model artefacts saved to models/:${NC}"
    echo "  🧠 bilstm_cancer_classifier.h5  — trained Keras model"
    echo "  📝 tokenizer.pkl               — fitted tokenizer"
    echo "  📊 training_history.png        — accuracy/loss curves"
    echo ""
    echo -e "${BOLD}Next step:${NC}"
    echo "  Step 3 → Run the full pipeline"
    echo "           python src/ingestion/loader.py"
    echo "  Or just tell Claude and I'll handle it!"
else
    fail "Some output files are missing. Check the training logs above for errors."
fi
