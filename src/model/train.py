"""
train.py — Train the Bidirectional LSTM cancer classifier from scratch.

Reproduces the IEEE-published model achieving ~99.8% accuracy on the
Kaggle Clinical Text Classification dataset (7,500+ records):
  - Thyroid Cancer (label 0)
  - Colon  Cancer  (label 1)
  - Lung   Cancer  (label 2)

Model architecture:
  Embedding → BiLSTM(128) → BiLSTM(64) → Dense(64) → Dropout → Dense(3, softmax)

Outputs saved to models/:
  - bilstm_cancer_classifier.h5  (trained Keras model)
  - tokenizer.pkl                (fitted Keras Tokenizer)
  - training_history.png         (accuracy / loss curves)

Usage:
    python src/model/train.py
    python src/model/train.py --epochs 15 --batch-size 64
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on all machines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# ── Add project root to path so src.* imports work ──────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    RAW_DATA_DIR,
    LABEL_MAP,
    MAX_SEQ_LEN,
    MODEL_PATH,
    TOKENIZER_PATH,
)

# ── Training hyper-parameters (also settable via CLI) ────────
DEFAULT_EPOCHS      = 10
DEFAULT_BATCH_SIZE  = 32
DEFAULT_VOCAB_SIZE  = 20_000    # top N most frequent words to keep
DEFAULT_EMBED_DIM   = 128       # word embedding dimension
DEFAULT_VAL_SPLIT   = 0.15      # 15% for validation
DEFAULT_TEST_SPLIT  = 0.15      # 15% for final test evaluation
RANDOM_SEED         = 42


def main(epochs: int, batch_size: int) -> None:
    """
    Full training pipeline:
      1. Load & validate the CSV dataset
      2. Preprocess text (same pipeline as preprocess.py)
      3. Tokenise and pad sequences
      4. Build the Bi-LSTM model
      5. Train with early stopping
      6. Evaluate on held-out test set
      7. Save model, tokenizer, and training plots
    """
    logger.info("=" * 60)
    logger.info("  cancer-pipeline  —  Bi-LSTM Training")
    logger.info("=" * 60)

    # ── 1. Load dataset ──────────────────────────────────────
    csv_path = RAW_DATA_DIR / "clinical_text.csv"
    df = _load_dataset(csv_path)

    # ── 2. Preprocess text ───────────────────────────────────
    logger.info("Preprocessing text...")
    df["clean_text"] = df["medical_abstract"].apply(_clean_text)
    logger.info(f"Sample clean text: {df['clean_text'].iloc[0][:80]}…")

    # ── 3. Tokenise & pad sequences ──────────────────────────
    X, tokenizer = _tokenise(df["clean_text"].tolist())
    y = _encode_labels(df["condition_label"].tolist())

    # Save tokenizer immediately — needed by the inference pipeline
    _save_tokenizer(tokenizer)

    # ── 4. Train / val / test split ──────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y)
    logger.info(
        f"Split → train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}"
    )

    # ── 5. Build model ───────────────────────────────────────
    model = _build_model(vocab_size=DEFAULT_VOCAB_SIZE, embed_dim=DEFAULT_EMBED_DIM)
    model.summary()

    # ── 6. Train ─────────────────────────────────────────────
    history = _train(model, X_train, y_train, X_val, y_val, epochs, batch_size)

    # ── 7. Evaluate on test set ──────────────────────────────
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test accuracy : {accuracy:.4%}")
    logger.info(f"Test loss     : {loss:.4f}")

    # ── 8. Save model & plots ────────────────────────────────
    _save_model(model)
    _save_plots(history)

    logger.info("=" * 60)
    logger.info(f"  Training complete!")
    logger.info(f"  Model saved     → {MODEL_PATH}")
    logger.info(f"  Tokenizer saved → {TOKENIZER_PATH}")
    logger.info(f"  Final accuracy  → {accuracy:.4%}")
    logger.info("=" * 60)


# ════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ════════════════════════════════════════════════════════════

def _load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load the Kaggle clinical text CSV and validate it.
    Raises FileNotFoundError if the CSV is not present.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset not found at: {csv_path}\n\n"
            "Please do ONE of the following:\n"
            "  Option A — Add your Kaggle credentials to .env:\n"
            "             KAGGLE_USERNAME=your_username\n"
            "             KAGGLE_KEY=your_api_key\n"
            "             Then run: python -c \"from src.ingestion.kaggle_downloader import download_dataset; download_dataset()\"\n\n"
            "  Option B — Download manually from Kaggle:\n"
            "             https://www.kaggle.com/datasets/ritheshsreenivasan/clinical-text-classification\n"
            "             Place the CSV at: data/raw/clinical_text.csv\n"
        )

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} records from {csv_path.name}")
    logger.info(f"Columns: {list(df.columns)}")

    # Validate required columns
    if "medical_abstract" not in df.columns or "condition_label" not in df.columns:
        raise ValueError(
            f"CSV must have 'medical_abstract' and 'condition_label' columns.\n"
            f"Found: {list(df.columns)}"
        )

    # Drop nulls
    before = len(df)
    df = df.dropna(subset=["medical_abstract", "condition_label"])
    df["condition_label"] = df["condition_label"].astype(int)
    logger.info(f"After cleaning: {len(df):,} records (dropped {before - len(df)} nulls)")

    # Show class distribution
    dist = df["condition_label"].map(LABEL_MAP).value_counts()
    logger.info(f"Class distribution:\n{dist.to_string()}")

    return df


def _clean_text(text: str) -> str:
    """
    NLP preprocessing — must match exactly what preprocess.py does
    so that training and inference use identical text representations.

    Steps: lowercase → remove non-alpha → collapse spaces → remove stop words
    """
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))

    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)


def _tokenise(texts: list[str]) -> tuple:
    """
    Fit a Keras Tokenizer on the training texts and convert all texts
    to padded integer sequences.

    Returns:
        padded_sequences: numpy array of shape (n_samples, MAX_SEQ_LEN)
        tokenizer:        fitted Keras Tokenizer (saved for inference)
    """
    from tensorflow.keras.preprocessing.text import Tokenizer       # type: ignore
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

    logger.info(f"Fitting tokenizer on {len(texts):,} texts (vocab size={DEFAULT_VOCAB_SIZE:,})...")
    tokenizer = Tokenizer(num_words=DEFAULT_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    word_count = len(tokenizer.word_index)
    logger.info(f"Unique words found: {word_count:,}")

    sequences = tokenizer.texts_to_sequences(texts)
    padded    = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="pre", truncating="pre")
    logger.info(f"Sequences padded to length {MAX_SEQ_LEN}. Shape: {padded.shape}")

    return padded, tokenizer


def _encode_labels(labels: list[int]):
    """One-hot encode labels for categorical cross-entropy loss."""
    from tensorflow.keras.utils import to_categorical  # type: ignore
    num_classes = len(LABEL_MAP)
    return to_categorical(labels, num_classes=num_classes)


def _split(X, y):
    """
    Stratified train / validation / test split.
    Stratified ensures each split has the same class proportions.
    """
    from sklearn.model_selection import train_test_split

    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=DEFAULT_TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y.argmax(axis=1),
    )
    # Then split remaining into train and validation
    val_ratio = DEFAULT_VAL_SPLIT / (1 - DEFAULT_TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=RANDOM_SEED,
        stratify=y_temp.argmax(axis=1),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _build_model(vocab_size: int, embed_dim: int):
    """
    Build the Bidirectional LSTM model architecture.

    Architecture (matching the IEEE-published design):
      ┌─────────────────────────────────────┐
      │  Embedding (vocab_size → embed_dim) │
      ├─────────────────────────────────────┤
      │  Bidirectional LSTM (128 units)     │  ← captures forward & backward context
      │  Dropout (0.3)                      │
      ├─────────────────────────────────────┤
      │  Bidirectional LSTM (64 units)      │  ← deeper feature extraction
      │  Dropout (0.3)                      │
      ├─────────────────────────────────────┤
      │  Dense (64, ReLU)                   │  ← classification head
      │  Dropout (0.5)                      │
      ├─────────────────────────────────────┤
      │  Dense (3, Softmax)                 │  ← 3 cancer classes
      └─────────────────────────────────────┘
    """
    import tensorflow as tf  # type: ignore

    model = tf.keras.Sequential([
        # Embedding: maps integer token IDs → dense vectors
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            input_length=MAX_SEQ_LEN,
            name="embedding",
        ),

        # First Bi-LSTM: return_sequences=True passes full sequence to next layer
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2),
            name="bilstm_1",
        ),
        tf.keras.layers.Dropout(0.3, name="dropout_1"),

        # Second Bi-LSTM: return_sequences=False — outputs single context vector
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2),
            name="bilstm_2",
        ),
        tf.keras.layers.Dropout(0.3, name="dropout_2"),

        # Classification head
        tf.keras.layers.Dense(64, activation="relu", name="dense_1"),
        tf.keras.layers.Dropout(0.5, name="dropout_3"),

        # Output layer: 3 neurons, softmax → probability per cancer type
        tf.keras.layers.Dense(3, activation="softmax", name="output"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def _train(model, X_train, y_train, X_val, y_val, epochs: int, batch_size: int):
    """
    Train with early stopping and best-model checkpointing.

    Callbacks:
      - EarlyStopping:    stop if val_accuracy doesn't improve for 3 epochs
      - ModelCheckpoint:  save the best weights automatically
      - ReduceLROnPlateau:reduce learning rate if val_loss plateaus
    """
    import tensorflow as tf  # type: ignore

    logger.info(f"Training for up to {epochs} epochs (batch size={batch_size})...")

    callbacks = [
        # Stop early if no improvement — prevents overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        # Always keep the best model on disk
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Reduce LR when learning stalls
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Ensure models/ directory exists
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def _save_tokenizer(tokenizer) -> None:
    """Pickle the fitted tokenizer so predictor.py can load it."""
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    logger.info(f"Tokenizer saved → {TOKENIZER_PATH}")


def _save_model(model) -> None:
    """Save the final model in .h5 format."""
    model.save(str(MODEL_PATH))
    size_mb = MODEL_PATH.stat().st_size / 1_048_576
    logger.info(f"Model saved → {MODEL_PATH}  ({size_mb:.1f} MB)")


def _save_plots(history) -> None:
    """Save accuracy and loss training curves as a PNG."""
    plots_path = MODEL_PATH.parent / "training_history.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax1.plot(history.history["accuracy"],     label="Train",      color="#2ecc71")
    ax1.plot(history.history["val_accuracy"], label="Validation", color="#e74c3c")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(history.history["loss"],     label="Train",      color="#2ecc71")
    ax2.plot(history.history["val_loss"], label="Validation", color="#e74c3c")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training plots saved → {plots_path}")


# ── CLI entry point ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Bi-LSTM cancer classifier")
    parser.add_argument("--epochs",     type=int, default=DEFAULT_EPOCHS,     help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    main(epochs=args.epochs, batch_size=args.batch_size)
