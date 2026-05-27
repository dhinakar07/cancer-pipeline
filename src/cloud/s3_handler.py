"""
s3_handler.py — AWS S3 utilities for cancer-pipeline.

Provides clean, logged wrappers around boto3 for:
  - Uploading raw CSV data to S3 after ingestion
  - Uploading processed data snapshots to S3
  - Downloading the Bi-LSTM model + tokenizer from S3 to local models/
  - Checking whether S3 objects exist (idempotent uploads)

Design principles:
  - Every function is safe to call even when S3 is not configured
    (it checks cloud_enabled() and logs a warning instead of crashing).
  - Uses multipart upload automatically for large files (boto3 default).
  - Credentials come from src/config.py → loaded from .env.
    In production (EC2 / MWAA), boto3 picks up the IAM Role automatically
    — no keys needed in the environment at all.

Usage:
    from src.cloud.s3_handler import upload_file, download_model_artefacts
"""

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path
from loguru import logger

from src.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    S3_BUCKET,
    S3_RAW_PREFIX,
    S3_PROCESSED_PREFIX,
    S3_MODEL_KEY,
    S3_TOKENIZER_KEY,
    MODEL_PATH,
    TOKENIZER_PATH,
    cloud_enabled,
)


# ── S3 client (singleton, created lazily) ───────────────────
_s3_client = None


def get_s3_client():
    """
    Return a (cached) boto3 S3 client.

    Credential priority (boto3 standard order):
      1. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in environment (.env)
      2. ~/.aws/credentials file (set up via `aws configure`)
      3. IAM Role attached to the EC2 / ECS / MWAA instance (production)

    Raises:
        RuntimeError: If no credentials are found.
    """
    global _s3_client
    if _s3_client is None:
        logger.info(f"Initialising S3 client in region '{AWS_REGION}'")
        try:
            # Pass explicit keys only if they are set; otherwise boto3
            # falls through to IAM role / ~/.aws/credentials automatically.
            kwargs = {"region_name": AWS_REGION}
            if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
                kwargs["aws_access_key_id"]     = AWS_ACCESS_KEY_ID
                kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY

            _s3_client = boto3.client("s3", **kwargs)
            logger.info("S3 client ready.")
        except NoCredentialsError:
            raise RuntimeError(
                "AWS credentials not found. Either:\n"
                "  1. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env, OR\n"
                "  2. Run `aws configure`, OR\n"
                "  3. Attach an IAM Role to your EC2 / MWAA instance."
            )
    return _s3_client


# ── Core upload / download helpers ──────────────────────────

def upload_file(local_path: Path, s3_key: str, bucket: str = S3_BUCKET) -> bool:
    """
    Upload a local file to S3.

    Args:
        local_path: Path to the file on disk.
        s3_key:     Destination key inside the S3 bucket (e.g. "data/raw/file.csv").
        bucket:     Target S3 bucket name (defaults to S3_BUCKET from config).

    Returns:
        True on success, False on failure.
    """
    if not cloud_enabled():
        logger.warning("S3 upload skipped — S3_BUCKET not configured in .env.")
        return False

    if not local_path.exists():
        logger.error(f"Upload failed: local file not found at {local_path}")
        return False

    try:
        client = get_s3_client()
        file_size_mb = local_path.stat().st_size / 1_048_576
        logger.info(f"Uploading {local_path.name} ({file_size_mb:.1f} MB) → s3://{bucket}/{s3_key}")

        client.upload_file(str(local_path), bucket, s3_key)
        logger.info(f"Upload complete: s3://{bucket}/{s3_key}")
        return True

    except ClientError as exc:
        logger.error(f"S3 upload failed: {exc}")
        return False


def download_file(s3_key: str, local_path: Path, bucket: str = S3_BUCKET) -> bool:
    """
    Download a file from S3 to a local path.

    Creates parent directories automatically.

    Args:
        s3_key:     Source key inside the S3 bucket.
        local_path: Destination path on disk.
        bucket:     Source S3 bucket name.

    Returns:
        True on success, False on failure.
    """
    if not cloud_enabled():
        logger.warning("S3 download skipped — S3_BUCKET not configured in .env.")
        return False

    try:
        client = get_s3_client()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading s3://{bucket}/{s3_key} → {local_path}")

        client.download_file(bucket, s3_key, str(local_path))
        logger.info(f"Download complete: {local_path}")
        return True

    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code == "404":
            logger.error(f"S3 object not found: s3://{bucket}/{s3_key}")
        else:
            logger.error(f"S3 download failed: {exc}")
        return False


def object_exists(s3_key: str, bucket: str = S3_BUCKET) -> bool:
    """
    Check whether an object exists in S3 without downloading it.

    Useful for idempotent uploads: skip if already present.

    Args:
        s3_key: Key to check inside the bucket.
        bucket: S3 bucket name.

    Returns:
        True if the object exists, False otherwise.
    """
    try:
        get_s3_client().head_object(Bucket=bucket, Key=s3_key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "404":
            return False
        raise   # unexpected error — re-raise for the caller to handle


# ── High-level pipeline helpers ─────────────────────────────

def upload_raw_data(local_csv: Path) -> bool:
    """
    Upload the raw Kaggle CSV to the S3 raw data prefix.

    S3 path: s3://<bucket>/data/raw/clinical_text.csv

    Args:
        local_csv: Local path to the downloaded CSV file.

    Returns:
        True on success, False if cloud is disabled or upload fails.
    """
    s3_key = f"{S3_RAW_PREFIX}{local_csv.name}"
    logger.info(f"Backing up raw data to s3://{S3_BUCKET}/{s3_key}")
    return upload_file(local_csv, s3_key)


def upload_processed_snapshot(local_csv: Path, run_id: str) -> bool:
    """
    Upload a processed data snapshot to S3, namespaced by run_id.

    S3 path: s3://<bucket>/data/processed/<run_id>/processed.csv

    Namespacing by run_id preserves a history of processed datasets
    across pipeline runs — useful for debugging prediction drift.

    Args:
        local_csv: Local path to the processed data CSV.
        run_id:    Airflow run_id (used as the S3 folder name).

    Returns:
        True on success, False if cloud is disabled or upload fails.
    """
    s3_key = f"{S3_PROCESSED_PREFIX}{run_id}/{local_csv.name}"
    logger.info(f"Uploading processed snapshot to s3://{S3_BUCKET}/{s3_key}")
    return upload_file(local_csv, s3_key)


def download_model_artefacts(force: bool = False) -> dict:
    """
    Download the Bi-LSTM model (.h5) and tokenizer (.pkl) from S3
    to the local models/ directory.

    If both files already exist locally and force=False, the download
    is skipped (safe to call on every pipeline run).

    Args:
        force: Re-download even if local files already exist.

    Returns:
        dict with keys "model_downloaded" and "tokenizer_downloaded" (bool each).
    """
    results = {"model_downloaded": False, "tokenizer_downloaded": False}

    if not cloud_enabled():
        logger.warning(
            "Model download from S3 skipped — S3_BUCKET not set.\n"
            f"Ensure the model is present locally at: {MODEL_PATH}"
        )
        return results

    # ── Download .h5 model ───────────────────────────────────
    if MODEL_PATH.exists() and not force:
        logger.info(f"Model already present locally at {MODEL_PATH} — skipping S3 download.")
    else:
        results["model_downloaded"] = download_file(S3_MODEL_KEY, MODEL_PATH)

    # ── Download tokenizer ───────────────────────────────────
    if TOKENIZER_PATH.exists() and not force:
        logger.info(f"Tokenizer already present at {TOKENIZER_PATH} — skipping S3 download.")
    else:
        results["tokenizer_downloaded"] = download_file(S3_TOKENIZER_KEY, TOKENIZER_PATH)

    return results


def upload_model_artefacts() -> dict:
    """
    Upload the local Bi-LSTM model and tokenizer to S3.

    Call this once after training to make artefacts available to all
    pipeline workers (e.g., multiple MWAA workers or EC2 instances).

    Returns:
        dict with keys "model_uploaded" and "tokenizer_uploaded" (bool each).
    """
    return {
        "model_uploaded":     upload_file(MODEL_PATH,     S3_MODEL_KEY),
        "tokenizer_uploaded": upload_file(TOKENIZER_PATH, S3_TOKENIZER_KEY),
    }


def list_bucket_objects(prefix: str = "", bucket: str = S3_BUCKET) -> list[str]:
    """
    List all object keys in the S3 bucket under a given prefix.

    Useful for debugging and verifying uploads.

    Args:
        prefix: S3 key prefix to filter by (e.g. "data/raw/").
        bucket: S3 bucket name.

    Returns:
        List of S3 key strings.
    """
    if not cloud_enabled():
        return []

    try:
        client   = get_s3_client()
        paginator = client.get_paginator("list_objects_v2")
        keys = []

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])

        logger.info(f"Found {len(keys)} objects under s3://{bucket}/{prefix}")
        return keys

    except ClientError as exc:
        logger.error(f"Failed to list S3 objects: {exc}")
        return []
