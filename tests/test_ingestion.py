"""
test_ingestion.py — Unit tests for src/ingestion/loader.py

Tests use a small in-memory CSV so no real database is required.
Database calls are mocked with unittest.mock.

Run with:
    pytest tests/test_ingestion.py -v
"""

import io
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# ── The module under test ────────────────────────────────────
from src.ingestion.loader import _read_and_validate, COL_TEXT, COL_LABEL


# ── Sample CSV content mimicking the Kaggle dataset ─────────
VALID_CSV = """medical_abstract,condition_label
"Patient presents with neck mass, thyroid nodule detected on ultrasound.",0
"Colonoscopy revealed polyp in the sigmoid colon with adenocarcinoma features.",1
"Chest CT shows solitary pulmonary nodule consistent with primary lung malignancy.",2
"Biopsy confirmed follicular thyroid carcinoma, staged T2N0M0.",0
"Lung adenocarcinoma with EGFR mutation identified in bronchoalveolar lavage.",2
"""

MISSING_COL_CSV = """text,label
"some text",0
"""

INVALID_LABEL_CSV = """medical_abstract,condition_label
"some medical text",99
"""


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def valid_csv_path(tmp_path):
    """Write the valid sample CSV to a temporary file and return its path."""
    p = tmp_path / "clinical_text.csv"
    p.write_text(VALID_CSV)
    return p


@pytest.fixture
def missing_col_csv_path(tmp_path):
    """Write a CSV with wrong column names."""
    p = tmp_path / "bad_columns.csv"
    p.write_text(MISSING_COL_CSV)
    return p


@pytest.fixture
def invalid_label_csv_path(tmp_path):
    """Write a CSV with an out-of-range label value."""
    p = tmp_path / "bad_labels.csv"
    p.write_text(INVALID_LABEL_CSV)
    return p


# ── Tests ────────────────────────────────────────────────────

class TestReadAndValidate:
    """Tests for the _read_and_validate() private helper."""

    def test_valid_csv_loads_all_rows(self, valid_csv_path):
        """Valid CSV should be loaded with 5 rows."""
        df = _read_and_validate(valid_csv_path)
        assert len(df) == 5

    def test_valid_csv_has_required_columns(self, valid_csv_path):
        """Loaded DataFrame must contain the required columns."""
        df = _read_and_validate(valid_csv_path)
        assert COL_TEXT  in df.columns
        assert COL_LABEL in df.columns

    def test_valid_csv_label_values(self, valid_csv_path):
        """All label values must be in {0, 1, 2}."""
        df = _read_and_validate(valid_csv_path)
        assert set(df[COL_LABEL].unique()).issubset({0, 1, 2})

    def test_missing_columns_raises_value_error(self, missing_col_csv_path):
        """CSV missing expected columns should raise ValueError."""
        with pytest.raises(ValueError, match="missing required columns"):
            _read_and_validate(missing_col_csv_path)

    def test_invalid_labels_raise_value_error(self, invalid_label_csv_path):
        """Labels outside {0, 1, 2} should raise ValueError."""
        with pytest.raises(ValueError, match="Unexpected label values"):
            _read_and_validate(invalid_label_csv_path)

    def test_nonexistent_file_raises_file_not_found(self, tmp_path):
        """A path that doesn't exist should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _read_and_validate(tmp_path / "does_not_exist.csv")

    def test_null_rows_are_dropped(self, tmp_path):
        """Rows with null text or label should be silently dropped."""
        csv_with_nulls = (
            "medical_abstract,condition_label\n"
            '"Valid text",0\n'
            ",1\n"          # null text → should be dropped
            '"More text",\n"# null label → should be dropped'
        )
        p = tmp_path / "nulls.csv"
        p.write_text(csv_with_nulls)
        df = _read_and_validate(p)
        assert len(df) == 1   # only the valid row survives

    def test_label_column_is_integer(self, valid_csv_path):
        """Label column must be cast to int, not float or string."""
        df = _read_and_validate(valid_csv_path)
        assert df[COL_LABEL].dtype == int
