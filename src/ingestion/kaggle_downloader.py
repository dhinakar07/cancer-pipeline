"""
kaggle_downloader.py — Download the cancer classification dataset from Kaggle.

Supports ALL three Kaggle credential formats:
  1. New token  — KAGGLE_API_TOKEN env var or ~/.kaggle/access_token (KGAT_xxx)
  2. Old API key — KAGGLE_USERNAME + KAGGLE_KEY in .env or environment
  3. kaggle.json — ~/.kaggle/kaggle.json (traditional format)
  4. Local file  — if CSV already in data/raw/, skip download entirely

Dataset: ritheshsreenivasan/clinical-text-classification
  7,500+ clinical text records labelled as:
    0 → Thyroid Cancer
    1 → Colon  Cancer
    2 → Lung   Cancer
"""

import os
import subprocess
import zipfile
from pathlib import Path

from loguru import logger

from src.config import KAGGLE_DATASET, RAW_DATA_DIR


# Expected CSV filename after extraction
EXPECTED_CSV = "clinical_text.csv"

# Path where the new Kaggle token is saved
KAGGLE_ACCESS_TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
KAGGLE_JSON_PATH          = Path.home() / ".kaggle" / "kaggle.json"


def _credentials_available() -> bool:
    """
    Return True if ANY valid Kaggle credential is found:
      - KAGGLE_API_TOKEN in environment       (new KGAT_xxx token)
      - ~/.kaggle/access_token file           (new KGAT_xxx token saved locally)
      - KAGGLE_USERNAME + KAGGLE_KEY in env   (old API key format)
      - ~/.kaggle/kaggle.json exists          (traditional JSON credentials)
    """
    return (
        bool(os.getenv("KAGGLE_API_TOKEN"))
        or KAGGLE_ACCESS_TOKEN_PATH.exists()
        or (bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY")))
        or KAGGLE_JSON_PATH.exists()
    )


def download_dataset(dest_dir: Path = RAW_DATA_DIR, force: bool = False) -> Path:
    """
    Download the Kaggle dataset to *dest_dir* and return the CSV path.

    If the CSV already exists and *force* is False, download is skipped
    (idempotent — safe to call on every pipeline run).

    Args:
        dest_dir: Directory where the raw data will be stored.
        force:    Re-download even if the file already exists.

    Returns:
        Path to the downloaded / existing CSV file.

    Raises:
        FileNotFoundError: No credentials and no local file found.
        RuntimeError:      Download or extraction failed.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / EXPECTED_CSV

    # ── Skip if file already present ────────────────────────
    if csv_path.exists() and not force:
        logger.info(f"Dataset already present at {csv_path} — skipping download.")
        return csv_path

    # ── Check credentials ────────────────────────────────────
    if not _credentials_available():
        raise FileNotFoundError(
            f"CSV not found at {csv_path} and no Kaggle credentials found.\n\n"
            "Fix: save your Kaggle token by running this command:\n"
            "  mkdir -p ~/.kaggle && echo YOUR_TOKEN > ~/.kaggle/access_token "
            "&& chmod 600 ~/.kaggle/access_token\n\n"
            "Or manually place the CSV at: data/raw/clinical_text.csv"
        )

    # ── Try CLI first (supports all credential formats) ─────
    logger.info(f"Downloading '{KAGGLE_DATASET}' via Kaggle CLI…")
    if _try_cli_download(dest_dir):
        pass
    else:
        # Fallback: use kaggle Python package API
        logger.info("CLI download failed, trying Python API…")
        _download_via_api(dest_dir)

    # ── Verify CSV exists after download ────────────────────
    # The dataset might have a different filename — rename if needed
    _ensure_csv_named_correctly(dest_dir, csv_path)

    if not csv_path.exists():
        raise RuntimeError(
            f"Download completed but CSV not found at {csv_path}.\n"
            f"Files in {dest_dir}: {list(dest_dir.iterdir())}"
        )

    logger.info(f"Dataset ready at {csv_path}")
    return csv_path


# ── Private helpers ──────────────────────────────────────────

def _try_cli_download(dest_dir: Path) -> bool:
    """
    Download using the kaggle CLI command.
    This supports all credential formats including the new KGAT_ token.

    Returns True on success, False on failure (so caller can fallback).
    """
    try:
        # Set KAGGLE_API_TOKEN from access_token file if not already set
        env = os.environ.copy()
        if not env.get("KAGGLE_API_TOKEN") and KAGGLE_ACCESS_TOKEN_PATH.exists():
            token = KAGGLE_ACCESS_TOKEN_PATH.read_text().strip()
            env["KAGGLE_API_TOKEN"] = token
            logger.info("Loaded Kaggle token from ~/.kaggle/access_token")

        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "--dataset", KAGGLE_DATASET,
                "--path",    str(dest_dir),
                "--unzip",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,   # 5-minute timeout
        )

        if result.returncode == 0:
            logger.info("Kaggle CLI download succeeded.")
            logger.debug(result.stdout)
            return True
        else:
            logger.warning(f"Kaggle CLI failed (exit {result.returncode}): {result.stderr}")
            return False

    except FileNotFoundError:
        logger.warning("kaggle CLI not found in PATH.")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Kaggle CLI download timed out after 5 minutes.")
        return False
    except Exception as exc:
        logger.warning(f"Kaggle CLI error: {exc}")
        return False


def _download_via_api(dest_dir: Path) -> None:
    """
    Fallback: use the kaggle Python package directly.
    Supports KAGGLE_USERNAME+KAGGLE_KEY and ~/.kaggle/kaggle.json.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended

        # Inject token from access_token file into env if needed
        if KAGGLE_ACCESS_TOKEN_PATH.exists() and not os.getenv("KAGGLE_API_TOKEN"):
            token = KAGGLE_ACCESS_TOKEN_PATH.read_text().strip()
            os.environ["KAGGLE_API_TOKEN"] = token

        api = KaggleApiExtended()
        api.authenticate()
        logger.info("Kaggle Python API authenticated.")

        api.dataset_download_files(
            dataset=KAGGLE_DATASET,
            path=str(dest_dir),
            unzip=False,
        )

        # Extract any downloaded zip files
        for zf_path in dest_dir.glob("*.zip"):
            logger.info(f"Extracting {zf_path.name}…")
            with zipfile.ZipFile(zf_path, "r") as zf:
                zf.extractall(dest_dir)
            zf_path.unlink()
            logger.info(f"Removed zip: {zf_path.name}")

    except ImportError:
        raise RuntimeError("kaggle package not installed. Run: pip install kaggle")
    except Exception as exc:
        raise RuntimeError(f"Kaggle API download failed: {exc}") from exc


def _ensure_csv_named_correctly(dest_dir: Path, expected_path: Path) -> None:
    """
    The Kaggle dataset may extract with a different filename.
    This function looks for any CSV in dest_dir and renames it
    to the expected name if needed.
    """
    if expected_path.exists():
        return  # already correct

    # Look for any CSV file in the directory
    csv_files = list(dest_dir.glob("*.csv"))
    if not csv_files:
        return  # nothing to rename — let caller handle the error

    if len(csv_files) == 1:
        # Only one CSV — rename it to the expected name
        csv_files[0].rename(expected_path)
        logger.info(f"Renamed {csv_files[0].name} → {expected_path.name}")
    else:
        # Multiple CSVs — log them so the user can identify the right one
        logger.warning(
            f"Multiple CSV files found in {dest_dir}:\n"
            + "\n".join(f"  {f.name}" for f in csv_files)
            + f"\nPlease rename the correct one to: {expected_path.name}"
        )
