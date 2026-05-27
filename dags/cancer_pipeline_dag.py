"""
cancer_pipeline_dag.py — Apache Airflow DAG for cancer-pipeline (cloud-enabled).

Orchestrates the full end-to-end pipeline on a daily schedule.
Works in both LOCAL and AWS CLOUD modes — no code changes needed,
just set S3_BUCKET and RDS_HOST in your .env (or Airflow Variables).

Pipeline flow:

  LOCAL mode (S3_BUCKET not set):
  ─────────────────────────────────────────────────────────────
  [download_dataset] → [load_raw_data] → [preprocess_text]
  → [run_predictions] → [update_run_metadata]

  CLOUD mode (S3_BUCKET configured):
  ─────────────────────────────────────────────────────────────
  [download_dataset] → [upload_raw_to_s3]
        ↓
  [load_raw_data] → [preprocess_text] → [upload_processed_to_s3]
        ↓
  [download_model_from_s3] → [run_predictions] → [update_run_metadata]

Author : dhinakaryalla07@gmail.com
Dataset: 7,500+ clinical records (Thyroid / Colon / Lung Cancer)
Model  : Bi-LSTM (99.8% accuracy, IEEE published)
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago

# ── Default arguments applied to every task ─────────────────
DEFAULT_ARGS = {
    "owner":            "dhinakar",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

# ── DAG definition ───────────────────────────────────────────
with DAG(
    dag_id="cancer_pipeline",
    description=(
        "End-to-end cancer classification pipeline (cloud-enabled): "
        "ingest → S3 backup → preprocess → Bi-LSTM predict → store results"
    ),
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["cancer", "nlp", "bi-lstm", "aws", "s3", "portfolio"],
) as dag:

    # ══════════════════════════════════════════════════════════
    # TASK 1: Download dataset from Kaggle (local → data/raw/)
    # ══════════════════════════════════════════════════════════
    def task_download(**context):
        """
        Download the clinical text CSV from Kaggle.
        Idempotent — skips if the file already exists in data/raw/.
        Pushes the CSV path and run_id to XCom for downstream tasks.
        """
        from src.ingestion.kaggle_downloader import download_dataset
        csv_path = download_dataset()
        context["ti"].xcom_push(key="csv_path",  value=str(csv_path))
        context["ti"].xcom_push(key="run_id",    value=context["run_id"])

    download_dataset_task = PythonOperator(
        task_id="download_dataset",
        python_callable=task_download,
    )

    # ══════════════════════════════════════════════════════════
    # TASK 2 (CLOUD): Upload raw CSV to S3
    # Skipped automatically when S3_BUCKET is not configured.
    # ══════════════════════════════════════════════════════════
    def task_upload_raw(**context):
        """
        Back up the raw CSV to S3 immediately after download.
        This ensures the source data is safely stored in the cloud
        even if the local machine crashes mid-pipeline.

        Uses ShortCircuitOperator logic: returns False (skip) when
        cloud storage is not configured, True to proceed.
        """
        from pathlib import Path
        from src.config import cloud_enabled
        from src.cloud.s3_handler import upload_raw_data

        if not cloud_enabled():
            # Returning False short-circuits and skips this task gracefully
            return False

        csv_path = context["ti"].xcom_pull(task_ids="download_dataset", key="csv_path")
        success  = upload_raw_data(Path(csv_path))
        context["ti"].xcom_push(key="raw_s3_uploaded", value=success)
        return True   # continue the pipeline regardless of upload result

    upload_raw_s3_task = ShortCircuitOperator(
        task_id="upload_raw_to_s3",
        python_callable=task_upload_raw,
        ignore_downstream_trigger_rules=False,   # don't skip downstream tasks
    )

    # ══════════════════════════════════════════════════════════
    # TASK 3: Load raw CSV into cancer.raw_records (PostgreSQL / RDS)
    # ══════════════════════════════════════════════════════════
    def task_load_raw(**context):
        """
        Truncate cancer.raw_records and bulk-insert all CSV rows.
        When RDS_HOST is set in .env, this writes to AWS RDS PostgreSQL.
        Pushes row count + label distribution to XCom.
        """
        from src.ingestion.loader import load_raw_data
        metrics = load_raw_data()
        context["ti"].xcom_push(key="ingest_metrics", value=metrics)

    load_raw_task = PythonOperator(
        task_id="load_raw_data",
        python_callable=task_load_raw,
    )

    # ══════════════════════════════════════════════════════════
    # TASK 4: NLP preprocessing → cancer.processed_records
    # ══════════════════════════════════════════════════════════
    def task_preprocess(**context):
        """
        Clean text (lowercase, remove stop words, etc.) and write
        results to cancer.processed_records (local or RDS).
        Exports a processed CSV snapshot for S3 upload.
        """
        import pandas as pd
        from src.transformation.preprocess import preprocess_records
        from src.config import PROCESSED_DATA_DIR
        from src.db.connection import get_engine
        from sqlalchemy import text

        metrics = preprocess_records()
        context["ti"].xcom_push(key="preprocess_metrics", value=metrics)

        # Export processed data to a local CSV for S3 backup
        run_id     = context["run_id"].replace(":", "-").replace("+", "-")
        engine     = get_engine()
        snap_path  = PROCESSED_DATA_DIR / f"processed_{run_id}.csv"
        snap_path.parent.mkdir(parents=True, exist_ok=True)

        with engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT * FROM cancer.processed_records"),
                conn,
            )
        df.to_csv(snap_path, index=False)
        context["ti"].xcom_push(key="processed_csv_path", value=str(snap_path))

    preprocess_task = PythonOperator(
        task_id="preprocess_text",
        python_callable=task_preprocess,
    )

    # ══════════════════════════════════════════════════════════
    # TASK 5 (CLOUD): Upload processed data snapshot to S3
    # Skipped when S3_BUCKET is not configured.
    # ══════════════════════════════════════════════════════════
    def task_upload_processed(**context):
        """
        Upload the processed data CSV snapshot to S3, namespaced by run_id.
        This preserves a versioned history of processed datasets in S3.
        """
        from pathlib import Path
        from src.config import cloud_enabled
        from src.cloud.s3_handler import upload_processed_snapshot

        if not cloud_enabled():
            return False

        csv_path = context["ti"].xcom_pull(
            task_ids="preprocess_text", key="processed_csv_path"
        )
        run_id  = context["run_id"]
        success = upload_processed_snapshot(Path(csv_path), run_id)
        context["ti"].xcom_push(key="processed_s3_uploaded", value=success)
        return True

    upload_processed_s3_task = ShortCircuitOperator(
        task_id="upload_processed_to_s3",
        python_callable=task_upload_processed,
        ignore_downstream_trigger_rules=False,
    )

    # ══════════════════════════════════════════════════════════
    # TASK 6 (CLOUD): Download Bi-LSTM model from S3
    # Skipped if model already exists locally or S3 not configured.
    # ══════════════════════════════════════════════════════════
    def task_download_model(**context):
        """
        Pull the trained Bi-LSTM .h5 model and tokenizer.pkl from S3
        to the local models/ directory.

        This makes the pipeline portable — any new worker (EC2, ECS,
        MWAA) can run predictions without pre-bundling the model files.

        Skips gracefully if:
          - S3 is not configured (local mode)
          - Files already exist locally (idempotent)
        """
        from src.config import cloud_enabled
        from src.cloud.s3_handler import download_model_artefacts

        if not cloud_enabled():
            return False   # local mode — model must already be in models/

        results = download_model_artefacts(force=False)
        context["ti"].xcom_push(key="model_download_results", value=results)
        return True

    download_model_task = ShortCircuitOperator(
        task_id="download_model_from_s3",
        python_callable=task_download_model,
        ignore_downstream_trigger_rules=False,
    )

    # ══════════════════════════════════════════════════════════
    # TASK 7: Run Bi-LSTM predictions → cancer.predictions
    # ══════════════════════════════════════════════════════════
    def task_predict(**context):
        """
        Load the Bi-LSTM model and score all processed records.
        Writes per-record predictions + softmax confidences to
        cancer.predictions (local PostgreSQL or AWS RDS).
        """
        from src.model.predictor import run_predictions
        run_id  = context["run_id"]
        metrics = run_predictions(pipeline_run_id=run_id)
        context["ti"].xcom_push(key="predict_metrics", value=metrics)

    predict_task = PythonOperator(
        task_id="run_predictions",
        python_callable=task_predict,
    )

    # ══════════════════════════════════════════════════════════
    # TASK 8: Write run summary to cancer.pipeline_runs
    # ══════════════════════════════════════════════════════════
    def task_update_metadata(**context):
        """
        Aggregate XCom metrics from all upstream tasks and upsert a
        complete run summary into cancer.pipeline_runs.
        This table is the source of truth for the Streamlit dashboard.
        """
        from src.db.connection import execute_sql
        from src.config import cloud_enabled

        ti = context["ti"]

        ingest_m     = ti.xcom_pull(task_ids="load_raw_data",     key="ingest_metrics")    or {}
        preprocess_m = ti.xcom_pull(task_ids="preprocess_text",   key="preprocess_metrics")or {}
        predict_m    = ti.xcom_pull(task_ids="run_predictions",   key="predict_metrics")   or {}

        run_id = context["run_id"]
        dag_id = context["dag"].dag_id

        # Tag whether this run used cloud storage for observability
        cloud_tag = "cloud" if cloud_enabled() else "local"
        dag_id_tagged = f"{dag_id}[{cloud_tag}]"

        upsert_sql = """
            INSERT INTO cancer.pipeline_runs
                (run_id, dag_id, status, records_ingested,
                 records_processed, records_predicted, accuracy, finished_at)
            VALUES
                (%s, %s, 'success', %s, %s, %s, %s, NOW())
            ON CONFLICT (run_id)
            DO UPDATE SET
                status             = EXCLUDED.status,
                records_ingested   = EXCLUDED.records_ingested,
                records_processed  = EXCLUDED.records_processed,
                records_predicted  = EXCLUDED.records_predicted,
                accuracy           = EXCLUDED.accuracy,
                finished_at        = EXCLUDED.finished_at;
        """
        execute_sql(upsert_sql, (
            run_id,
            dag_id_tagged,
            ingest_m.get("rows_inserted"),
            preprocess_m.get("rows_processed"),
            predict_m.get("rows_predicted"),
            predict_m.get("accuracy"),
        ))

    update_metadata_task = PythonOperator(
        task_id="update_run_metadata",
        python_callable=task_update_metadata,
        trigger_rule="all_done",   # run even if a task was skipped or failed
    )

    # ══════════════════════════════════════════════════════════
    # TASK DEPENDENCIES
    #
    # LOCAL  mode: 1 → 2(skip) → 3 → 4 → 5(skip) → 6(skip) → 7 → 8
    # CLOUD  mode: 1 → 2       → 3 → 4 → 5        → 6       → 7 → 8
    # ══════════════════════════════════════════════════════════
    (
        download_dataset_task
        >> upload_raw_s3_task
        >> load_raw_task
        >> preprocess_task
        >> upload_processed_s3_task
        >> download_model_task
        >> predict_task
        >> update_metadata_task
    )
