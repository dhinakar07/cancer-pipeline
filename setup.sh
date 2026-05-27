#!/bin/bash
# ============================================================
# setup.sh — One-click setup for cancer-pipeline (macOS / Linux)
#
# What this script does:
#   1. Checks that Docker Desktop is installed and running
#   2. Starts PostgreSQL via docker-compose
#   3. Waits until the database is healthy
#   4. Applies the schema (creates 4 tables in cancer schema)
#   5. Installs all Python packages from requirements.txt
#   6. Prints a success summary
#
# Usage:
#   chmod +x setup.sh   (only needed once)
#   ./setup.sh
# ============================================================

set -e   # stop immediately if any command fails

# ── Colours for readable output ─────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'   # No Colour

ok()   { echo -e "${GREEN}✅  $1${NC}"; }
info() { echo -e "${YELLOW}➜  $1${NC}"; }
fail() { echo -e "${RED}❌  $1${NC}"; exit 1; }

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   cancer-pipeline  —  Step 1 Setup Script   ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── Move to the project root (wherever this script lives) ───
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
info "Project directory: $PROJECT_DIR"

# ════════════════════════════════════════════════════════════
# STEP A — Check Docker is installed
# ════════════════════════════════════════════════════════════
info "Checking Docker installation..."

if ! command -v docker &> /dev/null; then
    fail "Docker is not installed.\nPlease install Docker Desktop from https://www.docker.com/products/docker-desktop/ and re-run this script."
fi
ok "Docker is installed: $(docker --version)"

# ════════════════════════════════════════════════════════════
# STEP B — Check Docker daemon is running
# ════════════════════════════════════════════════════════════
info "Checking Docker daemon..."

if ! docker info &> /dev/null; then
    fail "Docker is installed but not running.\nPlease open Docker Desktop, wait for it to start (whale icon in menu bar), then re-run this script."
fi
ok "Docker daemon is running."

# ════════════════════════════════════════════════════════════
# STEP C — Check .env file exists
# ════════════════════════════════════════════════════════════
info "Checking .env file..."

if [ ! -f ".env" ]; then
    info ".env not found — copying from .env.example..."
    cp .env.example .env
    ok ".env created from .env.example."
else
    ok ".env already exists."
fi

# ════════════════════════════════════════════════════════════
# STEP D — Start PostgreSQL with docker-compose
# ════════════════════════════════════════════════════════════
info "Starting PostgreSQL with docker-compose..."

# Use 'docker compose' (v2) or 'docker-compose' (v1) — whichever is available
if docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi
ok "Using compose command: $COMPOSE_CMD"

$COMPOSE_CMD up -d postgres
ok "PostgreSQL container started."

# ════════════════════════════════════════════════════════════
# STEP E — Wait for PostgreSQL to be healthy (max 60 seconds)
# ════════════════════════════════════════════════════════════
info "Waiting for PostgreSQL to be ready..."

MAX_WAIT=60
WAITED=0
until docker exec cancer_pipeline_db pg_isready -U cancer_user -d cancer_pipeline &> /dev/null; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        fail "PostgreSQL did not become ready within ${MAX_WAIT}s.\nCheck logs with: docker logs cancer_pipeline_db"
    fi
    echo -ne "   Waiting... ${WAITED}s\r"
    sleep 2
    WAITED=$((WAITED + 2))
done
ok "PostgreSQL is healthy and accepting connections."

# ════════════════════════════════════════════════════════════
# STEP F — Apply the database schema
# ════════════════════════════════════════════════════════════
info "Applying database schema..."

# Run schema.sql inside the container (no need for psql on the host machine)
docker exec -i cancer_pipeline_db \
    psql -U cancer_user -d cancer_pipeline \
    < src/db/schema.sql

ok "Schema applied."

# Verify the 4 tables were created
TABLE_COUNT=$(docker exec cancer_pipeline_db \
    psql -U cancer_user -d cancer_pipeline -t \
    -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'cancer';" \
    | tr -d ' ')

if [ "$TABLE_COUNT" -ge 4 ]; then
    ok "Database has ${TABLE_COUNT} tables in the 'cancer' schema."
else
    fail "Expected 4 tables but found ${TABLE_COUNT}. Check schema.sql."
fi

# ════════════════════════════════════════════════════════════
# STEP G — Install Python packages
# ════════════════════════════════════════════════════════════
info "Installing Python packages from requirements.txt..."
echo "   (This may take 2-3 minutes — grabbing 22 packages including TensorFlow)"
echo ""

# Try pip3 first, then pip
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    fail "pip is not installed. Please install Python 3.11+ from https://python.org"
fi

$PIP_CMD install -r requirements.txt --quiet
ok "All Python packages installed successfully."

# ════════════════════════════════════════════════════════════
# STEP H — Quick Python smoke test
# ════════════════════════════════════════════════════════════
info "Running quick smoke test..."

python3 -c "
import pandas, numpy, sqlalchemy, psycopg2, loguru, dotenv
print('  Core imports: OK')
" && ok "Python imports working correctly."

# ════════════════════════════════════════════════════════════
# DONE — Print summary
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║            Step 1 Complete! 🎉               ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}What's running:${NC}"
echo "  🐘 PostgreSQL  → localhost:5432  (DB: cancer_pipeline)"
echo "  📦 22 Python packages installed"
echo "  🗄️  4 tables created in the 'cancer' schema"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo "  Step 2 → Copy your model files into models/"
echo "           models/bilstm_cancer_classifier.h5"
echo "           models/tokenizer.pkl"
echo ""
echo -e "${BOLD}Useful commands:${NC}"
echo "  Check DB tables:  docker exec -it cancer_pipeline_db psql -U cancer_user -d cancer_pipeline -c '\dt cancer.*'"
echo "  Stop DB:          $COMPOSE_CMD down"
echo "  View DB logs:     docker logs cancer_pipeline_db"
echo ""
