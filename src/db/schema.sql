-- ============================================================
-- schema.sql — PostgreSQL schema for cancer-pipeline
-- Run once to initialise the database:
--   psql -U cancer_user -d cancer_pipeline -f src/db/schema.sql
-- Docker Compose mounts this file and runs it automatically.
-- ============================================================

-- Use a dedicated schema to keep things organised
CREATE SCHEMA IF NOT EXISTS cancer;

-- ── Raw medical text records ─────────────────────────────────
-- Stores the original records exactly as ingested from the
-- Kaggle "Clinical Text Classification" dataset.
CREATE TABLE IF NOT EXISTS cancer.raw_records (
    id              SERIAL PRIMARY KEY,
    source_file     TEXT        NOT NULL,           -- original CSV filename
    original_text   TEXT        NOT NULL,           -- raw medical text
    label_raw       INTEGER     NOT NULL,           -- 0=Thyroid, 1=Colon, 2=Lung
    label_name      TEXT        NOT NULL,           -- human-readable label
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Preprocessed / cleaned text ─────────────────────────────
-- Stores text after NLP preprocessing (lowercasing, stop-word
-- removal, etc.) ready for the Bi-LSTM model.
CREATE TABLE IF NOT EXISTS cancer.processed_records (
    id              SERIAL PRIMARY KEY,
    raw_record_id   INTEGER     NOT NULL REFERENCES cancer.raw_records(id) ON DELETE CASCADE,
    clean_text      TEXT        NOT NULL,           -- preprocessed text
    token_count     INTEGER,                        -- number of tokens after cleaning
    processed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Model predictions ────────────────────────────────────────
-- One row per record per pipeline run; supports tracking
-- prediction drift over time by keeping historical runs.
CREATE TABLE IF NOT EXISTS cancer.predictions (
    id                  SERIAL PRIMARY KEY,
    processed_record_id INTEGER     NOT NULL REFERENCES cancer.processed_records(id) ON DELETE CASCADE,
    pipeline_run_id     TEXT        NOT NULL,       -- Airflow run_id for traceability
    predicted_label     INTEGER     NOT NULL,       -- 0 / 1 / 2
    predicted_name      TEXT        NOT NULL,       -- Thyroid / Colon / Lung
    confidence_thyroid  NUMERIC(6,4),               -- softmax probability for class 0
    confidence_colon    NUMERIC(6,4),               -- softmax probability for class 1
    confidence_lung     NUMERIC(6,4),               -- softmax probability for class 2
    is_correct          BOOLEAN,                    -- NULL until ground truth is compared
    predicted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Pipeline run metadata ────────────────────────────────────
-- One row per Airflow DAG run; useful for monitoring & audit.
CREATE TABLE IF NOT EXISTS cancer.pipeline_runs (
    run_id          TEXT        PRIMARY KEY,        -- Airflow run_id
    dag_id          TEXT        NOT NULL,
    status          TEXT        NOT NULL,           -- running / success / failed
    records_ingested INTEGER,
    records_processed INTEGER,
    records_predicted INTEGER,
    accuracy        NUMERIC(6,4),                   -- accuracy vs. ground truth (if available)
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ
);

-- ── Indexes for common query patterns ───────────────────────
CREATE INDEX IF NOT EXISTS idx_raw_label       ON cancer.raw_records(label_raw);
CREATE INDEX IF NOT EXISTS idx_pred_run        ON cancer.predictions(pipeline_run_id);
CREATE INDEX IF NOT EXISTS idx_pred_label      ON cancer.predictions(predicted_label);
CREATE INDEX IF NOT EXISTS idx_pred_correct    ON cancer.predictions(is_correct);
CREATE INDEX IF NOT EXISTS idx_run_status      ON cancer.pipeline_runs(status);
