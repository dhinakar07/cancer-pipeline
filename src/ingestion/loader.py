"""
loader.py — Ingest raw CSV data into the cancer.raw_records PostgreSQL table.

Pipeline step 1: Read CSV → validate → bulk-insert into Postgres.

Key design decisions:
  - Uses pandas + SQLAlchemy for a clean, readable insert path.
  - Validates required columns and label values before touching the DB.
  - Idempotent: truncates the staging table before each load so re-runs
    don't accumulate duplicates (full-refresh strategy).
  - Returns a metrics dict so the Airflow task can log row counts.
"""

import pandas as pd
from pathlib import Path
from loguru import logger

from src.config import LABEL_MAP, RAW_DATA_DIR
from src.db.connection import get_engine, execute_sql

# ── Column name constants (match the Kaggle CSV headers) ────
COL_TEXT  = "medical_abstract"   # raw clinical text column
COL_LABEL = "condition_label"    # integer label column (0 / 1 / 2)


def load_raw_data(csv_path: Path | None = None) -> dict:
    """
    Read the Kaggle CSV, validate it, and insert all records into
    cancer.raw_records (after truncating the table for a fresh load).

    Args:
        csv_path: Path to the CSV. Defaults to data/raw/clinical_text.csv.

    Returns:
        dict with keys:
            "rows_read"     – rows in the CSV
            "rows_inserted" – rows successfully inserted into Postgres
            "label_counts"  – {label_name: count} distribution
    """
    if csv_path is None:
        csv_path = RAW_DATA_DIR / "clinical_text.csv"

    logger.info(f"Reading CSV from {csv_path}")
    df = _read_and_validate(csv_path)

    logger.info(f"Truncating cancer.raw_records for fresh load…")
    execute_sql("TRUNCATE TABLE cancer.raw_records RESTART IDENTITY CASCADE;")

    rows_inserted = _insert_records(df)
    label_counts  = df[COL_LABEL].map(LABEL_MAP).value_counts().to_dict()

    metrics = {
        "rows_read":     len(df),
        "rows_inserted": rows_inserted,
        "label_counts":  label_counts,
    }
    logger.info(f"Ingestion complete: {metrics}")
    return metrics


# ── Private helpers ─────────────────────────────────────────

def _read_and_validate(csv_path: Path) -> pd.DataFrame:
    """
    Read the CSV and assert it has the expected columns and label values.

    Raises:
        FileNotFoundError: CSV does not exist.
        ValueError:        Required columns missing or labels out of range.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

    # ── Column presence check ───────────────────────────────
    required_cols = {COL_TEXT, COL_LABEL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # ── Drop rows with null text or label ───────────────────
    before = len(df)
    df = df.dropna(subset=[COL_TEXT, COL_LABEL])
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with null text/label.")

    # ── Label range validation ───────────────────────────────
    valid_labels = set(LABEL_MAP.keys())          # {0, 1, 2}
    df[COL_LABEL] = df[COL_LABEL].astype(int)
    invalid = set(df[COL_LABEL].unique()) - valid_labels
    if invalid:
        raise ValueError(
            f"Unexpected label values: {invalid}. Expected one of {valid_labels}."
        )

    return df


def _insert_records(df: pd.DataFrame) -> int:
    """
    Bulk-insert the validated DataFrame into cancer.raw_records.

    Builds the insert DataFrame with the exact columns the table expects
    and uses pandas.to_sql for an efficient multi-row insert.

    Returns:
        Number of rows inserted.
    """
    # Map integer label → human name
    insert_df = pd.DataFrame({
        "source_file":   "clinical_text.csv",
        "original_text": df[COL_TEXT].str.strip(),
        "label_raw":     df[COL_LABEL].astype(int),
        "label_name":    df[COL_LABEL].map(LABEL_MAP),
    })

    engine = get_engine()
    insert_df.to_sql(
        name="raw_records",
        con=engine,
        schema="cancer",
        if_exists="append",   # table was already TRUNCATEd above
        index=False,
        method="multi",       # sends multiple rows per INSERT statement
        chunksize=500,        # 500 rows per batch for memory efficiency
    )

    logger.info(f"Inserted {len(insert_df):,} rows into cancer.raw_records.")
    return len(insert_df)
