"""
predictor.py — Run the Bi-LSTM cancer classifier and write predictions to Postgres.

Pipeline step 3: Load saved Keras model → tokenise processed text →
run inference → store per-record predictions in cancer.predictions.

The IEEE-published Bi-LSTM model achieves 99.8% accuracy across three
cancer types: Thyroid (0), Colon (1), Lung (2).

Model artefacts expected in the models/ directory:
  - bilstm_cancer_classifier.h5  (or a SavedModel folder)
  - tokenizer.pkl                (fitted Keras Tokenizer, pickled)
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.config import MODEL_PATH, TOKENIZER_PATH, MAX_SEQ_LEN, LABEL_MAP
from src.db.connection import get_engine, execute_sql
from src.transformation.preprocess import get_text_sequences


def run_predictions(pipeline_run_id: str) -> dict:
    """
    Load the Bi-LSTM model, predict on all processed records, and
    persist the results to cancer.predictions.

    Args:
        pipeline_run_id: Airflow run_id string (for traceability).

    Returns:
        dict with keys:
            "rows_predicted" – number of predictions written
            "accuracy"       – fraction correct vs. ground-truth labels
                               (None if ground truth is unavailable)
    """
    engine = get_engine()

    # ── Load processed records + ground-truth labels ────────
    logger.info("Loading processed records for prediction…")
    df = pd.read_sql(
        """
        SELECT
            pr.id           AS processed_record_id,
            pr.clean_text,
            rr.label_raw    AS true_label
        FROM cancer.processed_records pr
        JOIN cancer.raw_records rr ON rr.id = pr.raw_record_id
        ORDER BY pr.id
        """,
        con=engine,
    )
    logger.info(f"Loaded {len(df):,} records for inference.")

    # ── Load the Keras model ─────────────────────────────────
    model = _load_model()

    # ── Tokenise & pad text ──────────────────────────────────
    logger.info("Tokenising and padding sequences…")
    sequences = get_text_sequences(df, TOKENIZER_PATH, MAX_SEQ_LEN)

    # ── Run inference ────────────────────────────────────────
    logger.info("Running Bi-LSTM inference…")
    # model.predict returns shape (n_samples, 3) softmax probabilities
    probabilities = model.predict(sequences, batch_size=64, verbose=0)

    # ── Build predictions DataFrame ──────────────────────────
    pred_labels  = np.argmax(probabilities, axis=1)   # class with highest prob
    pred_df = pd.DataFrame({
        "processed_record_id": df["processed_record_id"].values,
        "pipeline_run_id":     pipeline_run_id,
        "predicted_label":     pred_labels,
        "predicted_name":      [LABEL_MAP[int(l)] for l in pred_labels],
        "confidence_thyroid":  np.round(probabilities[:, 0], 4),
        "confidence_colon":    np.round(probabilities[:, 1], 4),
        "confidence_lung":     np.round(probabilities[:, 2], 4),
        "is_correct":          (pred_labels == df["true_label"].values),
    })

    # ── Accuracy metric ──────────────────────────────────────
    accuracy = float((pred_df["is_correct"]).mean())
    logger.info(f"Accuracy vs. ground truth: {accuracy:.4%}")

    # ── Persist to Postgres ──────────────────────────────────
    # Delete any previous predictions for this run (idempotent re-runs)
    execute_sql(
        "DELETE FROM cancer.predictions WHERE pipeline_run_id = %s",
        (pipeline_run_id,),
    )

    pred_df.to_sql(
        name="predictions",
        con=engine,
        schema="cancer",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500,
    )
    logger.info(
        f"Stored {len(pred_df):,} predictions for run '{pipeline_run_id}'."
    )

    return {
        "rows_predicted": len(pred_df),
        "accuracy":       round(accuracy, 4),
    }


# ── Private helpers ─────────────────────────────────────────

def _load_model():
    """
    Load the saved Bi-LSTM Keras model from MODEL_PATH.

    Cloud-aware: if the model file is not present locally AND S3 is
    configured, it automatically downloads it from S3 first.

    Supports both:
      - .h5 file (legacy HDF5 format)
      - SavedModel directory

    Raises:
        FileNotFoundError: If MODEL_PATH does not exist locally
                           and cannot be downloaded from S3.
    """
    import tensorflow as tf   # imported lazily to keep startup fast

    # ── Try to download from S3 if not present locally ──────
    if not MODEL_PATH.exists():
        logger.info(
            f"Model not found locally at {MODEL_PATH}. "
            "Attempting to download from S3…"
        )
        # Import here to avoid circular imports and keep startup fast
        from src.cloud.s3_handler import download_model_artefacts
        result = download_model_artefacts()

        if not result.get("model_downloaded"):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH} and S3 download failed.\n"
                "Either:\n"
                "  1. Copy bilstm_cancer_classifier.h5 to the models/ folder, OR\n"
                "  2. Configure S3_BUCKET in .env and upload the model with:\n"
                "     python -c 'from src.cloud.s3_handler import upload_model_artefacts; upload_model_artefacts()'"
            )

    logger.info(f"Loading Bi-LSTM model from {MODEL_PATH}…")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    logger.info(f"Model loaded. Input shape: {model.input_shape}")
    return model
