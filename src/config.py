"""
config.py — Centralised configuration for cancer-pipeline.

All environment variables are loaded once here and exposed as
typed constants so every other module imports from this file
instead of calling os.getenv() directly.

Usage:
    from src.config import DB_URL, MODEL_PATH, MAX_SEQ_LEN
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file (if present) ─────────────────────────────
# This is a no-op when running inside Docker/Airflow because
# the variables are already injected into the environment.
load_dotenv()

# ── Project root (one level above this file) ────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── PostgreSQL ───────────────────────────────────────────────
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB   = os.getenv("POSTGRES_DB", "cancer_pipeline")
POSTGRES_USER = os.getenv("POSTGRES_USER", "cancer_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

# SQLAlchemy connection string (used by pandas and SQLAlchemy)
DB_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# psycopg2 dict (used for raw cursor connections)
DB_CONN_PARAMS = {
    "host":     POSTGRES_HOST,
    "port":     POSTGRES_PORT,
    "dbname":   POSTGRES_DB,
    "user":     POSTGRES_USER,
    "password": POSTGRES_PASSWORD,
}

# ── Kaggle ───────────────────────────────────────────────────
KAGGLE_DATASET = os.getenv(
    "KAGGLE_DATASET", "ritheshsreenivasan/clinical-text-classification"
)

# ── Data paths ───────────────────────────────────────────────
RAW_DATA_DIR       = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# ── Model / tokenizer ────────────────────────────────────────
MODEL_PATH     = Path(os.getenv("MODEL_PATH",     str(ROOT_DIR / "models" / "bilstm_cancer_classifier.h5")))
TOKENIZER_PATH = Path(os.getenv("TOKENIZER_PATH", str(ROOT_DIR / "models" / "tokenizer.pkl")))
MAX_SEQ_LEN    = int(os.getenv("MAX_SEQUENCE_LENGTH", 500))

# ── Cancer label mapping (from the Kaggle dataset) ───────────
# The dataset encodes labels as integers; map back to human names.
LABEL_MAP: dict[int, str] = {
    0: "Thyroid Cancer",
    1: "Colon Cancer",
    2: "Lung Cancer",
}

# ── Cancer label mapping (from the Kaggle dataset) ───────────
# The dataset encodes labels as integers; map back to human names.

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ════════════════════════════════════════════════════════════
# AWS SETTINGS
# Set these in .env to enable cloud storage (S3) and managed
# PostgreSQL (RDS). The pipeline works fully locally without them.
# ════════════════════════════════════════════════════════════

# ── AWS credentials & region ─────────────────────────────────
# Prefer IAM roles (EC2/ECS/MWAA) over hard-coded keys in production.
# For local dev, set these in .env or configure via `aws configure`.
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID",     "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION            = os.getenv("AWS_REGION",            "us-east-1")

# ── S3 bucket & key prefixes ─────────────────────────────────
S3_BUCKET             = os.getenv("S3_BUCKET", "")           # e.g. my-cancer-pipeline
S3_RAW_PREFIX         = os.getenv("S3_RAW_PREFIX",       "data/raw/")
S3_PROCESSED_PREFIX   = os.getenv("S3_PROCESSED_PREFIX", "data/processed/")
S3_MODEL_PREFIX       = os.getenv("S3_MODEL_PREFIX",     "models/")

# Full S3 keys for model artefacts
S3_MODEL_KEY          = f"{S3_MODEL_PREFIX}bilstm_cancer_classifier.h5"
S3_TOKENIZER_KEY      = f"{S3_MODEL_PREFIX}tokenizer.pkl"

# ── RDS (cloud-managed PostgreSQL) ───────────────────────────
# When set, these override the local POSTGRES_HOST/PORT above.
# Leave blank to use the local Docker PostgreSQL.
RDS_HOST = os.getenv("RDS_HOST", "")    # e.g. mydb.xxxx.us-east-1.rds.amazonaws.com
RDS_PORT = int(os.getenv("RDS_PORT", POSTGRES_PORT))

# If RDS_HOST is set, override the DB connection with RDS endpoint
if RDS_HOST:
    POSTGRES_HOST = RDS_HOST
    POSTGRES_PORT = RDS_PORT
    DB_URL = (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    DB_CONN_PARAMS["host"] = POSTGRES_HOST
    DB_CONN_PARAMS["port"] = POSTGRES_PORT

# ── Helper: check if cloud storage is configured ─────────────
def cloud_enabled() -> bool:
    """Return True if an S3 bucket name has been configured."""
    return bool(S3_BUCKET)
