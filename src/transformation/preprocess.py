"""
preprocess.py — NLP preprocessing for cancer-pipeline.

Pipeline step 2: Read cancer.raw_records → clean text → write to
cancer.processed_records.

Preprocessing steps (matching what was used during Bi-LSTM training):
  1. Lowercase
  2. Remove non-alphabetic characters
  3. Tokenise on whitespace
  4. Remove NLTK English stop words
  5. Rejoin tokens → clean_text string
  6. Compute token_count for monitoring

Returns a metrics dict for Airflow XCom logging.
"""

import re
import pickle
from pathlib import Path

import pandas as pd
import nltk
from loguru import logger
from nltk.corpus import stopwords

from src.db.connection import get_engine, execute_sql

# ── Download NLTK stop-word list if not already cached ──────
nltk.download("stopwords", quiet=True)
_STOP_WORDS = set(stopwords.words("english"))


def preprocess_records() -> dict:
    """
    Read all rows from cancer.raw_records, apply NLP cleaning, and
    insert the results into cancer.processed_records.

    The processed_records table is truncated before each run (full-refresh).

    Returns:
        dict with keys:
            "rows_processed"   – total rows cleaned
            "avg_token_count"  – mean token count after cleaning
    """
    engine = get_engine()

    # ── Load raw records from Postgres ──────────────────────
    logger.info("Loading raw records from cancer.raw_records…")
    df = pd.read_sql(
        "SELECT id, original_text FROM cancer.raw_records ORDER BY id",
        con=engine,
    )
    logger.info(f"Loaded {len(df):,} raw records.")

    # ── Apply cleaning ───────────────────────────────────────
    logger.info("Cleaning text…")
    df["clean_text"]   = df["original_text"].apply(_clean_text)
    df["token_count"]  = df["clean_text"].apply(lambda t: len(t.split()))

    # ── Truncate target table & insert ──────────────────────
    logger.info("Truncating cancer.processed_records…")
    execute_sql(
        "TRUNCATE TABLE cancer.processed_records RESTART IDENTITY CASCADE;"
    )

    insert_df = pd.DataFrame({
        "raw_record_id": df["id"],
        "clean_text":    df["clean_text"],
        "token_count":   df["token_count"],
    })
    insert_df.to_sql(
        name="processed_records",
        con=engine,
        schema="cancer",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500,
    )
    logger.info(f"Inserted {len(insert_df):,} rows into cancer.processed_records.")

    metrics = {
        "rows_processed":  len(insert_df),
        "avg_token_count": round(float(df["token_count"].mean()), 2),
    }
    logger.info(f"Preprocessing complete: {metrics}")
    return metrics


def _clean_text(text: str) -> str:
    """
    Apply the same preprocessing pipeline used during Bi-LSTM training.

    Steps:
      1. Lowercase the text.
      2. Remove everything that is not a letter or a space.
      3. Collapse multiple spaces into one.
      4. Remove stop words.
      5. Return the cleaned string (empty string if nothing remains).

    Args:
        text: Raw medical abstract string.

    Returns:
        Cleaned, stop-word-free string.
    """
    if not isinstance(text, str):
        return ""

    # Step 1 – lowercase
    text = text.lower()

    # Step 2 – keep only letters and spaces (removes digits, punctuation, etc.)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 3 – normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Step 4 – remove stop words
    tokens = [word for word in text.split() if word not in _STOP_WORDS]

    return " ".join(tokens)


def get_text_sequences(
    df: pd.DataFrame,
    tokenizer_path: Path,
    max_len: int,
) -> "list[list[int]]":
    """
    Convert clean text into padded integer sequences for the Bi-LSTM model.

    This function is used by src/model/predictor.py after preprocessing.

    Args:
        df:             DataFrame with a 'clean_text' column.
        tokenizer_path: Path to the pickled Keras Tokenizer.
        max_len:        Sequence length used during model training.

    Returns:
        2-D list of shape (n_samples, max_len) with integer token IDs.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

    # Load the tokenizer that was fitted during model training
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
        logger.info(f"Tokenizer loaded from {tokenizer_path}")

    # Convert texts → sequences of integers
    sequences = tokenizer.texts_to_sequences(df["clean_text"].tolist())

    # Pad / truncate to fixed length (pre-padding matches training convention)
    padded = pad_sequences(sequences, maxlen=max_len, padding="pre", truncating="pre")
    return padded
