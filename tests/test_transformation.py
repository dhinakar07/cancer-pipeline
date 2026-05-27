"""
test_transformation.py — Unit tests for src/transformation/preprocess.py

Tests focus on the _clean_text() function which must match the exact
preprocessing applied during Bi-LSTM model training.

Run with:
    pytest tests/test_transformation.py -v
"""

import pytest
from src.transformation.preprocess import _clean_text


class TestCleanText:
    """Tests for the _clean_text() NLP preprocessing function."""

    def test_basic_lowercase(self):
        """Text should be fully lowercased."""
        result = _clean_text("THYROID CANCER DIAGNOSIS")
        assert result == result.lower()

    def test_removes_punctuation(self):
        """Punctuation, digits, and special characters should be removed."""
        result = _clean_text("Patient, 45-years-old! Diagnosed: 99% certain.")
        assert "," not in result
        assert "." not in result
        assert "!" not in result
        assert "%" not in result

    def test_removes_digits(self):
        """Numeric characters should be stripped."""
        result = _clean_text("T2N0M0 staging 2024")
        assert not any(ch.isdigit() for ch in result)

    def test_removes_stop_words(self):
        """Common English stop words should be removed."""
        result = _clean_text("the patient is a confirmed case of lung cancer")
        # 'the', 'is', 'a', 'of' are stop words
        tokens = result.split()
        assert "the" not in tokens
        assert "is"  not in tokens
        assert "a"   not in tokens
        assert "of"  not in tokens

    def test_meaningful_words_preserved(self):
        """Domain-specific medical terms should survive preprocessing."""
        result = _clean_text("thyroid nodule biopsy adenocarcinoma")
        assert "thyroid"       in result
        assert "nodule"        in result
        assert "biopsy"        in result
        assert "adenocarcinoma" in result

    def test_empty_string_returns_empty(self):
        """An empty input should produce an empty output."""
        assert _clean_text("") == ""

    def test_none_input_returns_empty(self):
        """None input should be handled gracefully and return empty string."""
        assert _clean_text(None) == ""

    def test_non_string_input_returns_empty(self):
        """Non-string inputs (int, float) should return empty string."""
        assert _clean_text(123)   == ""
        assert _clean_text(3.14)  == ""

    def test_only_stop_words_returns_empty(self):
        """A string containing only stop words should result in empty string."""
        result = _clean_text("the a is of and or")
        assert result.strip() == ""

    def test_whitespace_normalised(self):
        """Multiple consecutive spaces should be collapsed into one."""
        result = _clean_text("colon   cancer   screening")
        assert "  " not in result   # no double spaces

    def test_output_is_string(self):
        """The return type should always be str."""
        assert isinstance(_clean_text("any text"), str)
        assert isinstance(_clean_text(""), str)
        assert isinstance(_clean_text(None), str)

    def test_real_medical_text(self):
        """
        Smoke test with a realistic clinical abstract.
        Verifies that meaningful tokens survive and noise is removed.
        """
        abstract = (
            "Patient is a 62-year-old female presenting with a 3.2 cm "
            "pulmonary nodule in the right upper lobe. CT-guided biopsy "
            "confirmed non-small cell lung carcinoma (NSCLC) with EGFR "
            "exon 19 deletion."
        )
        result = _clean_text(abstract)

        # Important medical terms should survive
        assert "pulmonary" in result
        assert "nodule"    in result
        assert "biopsy"    in result
        assert "lung"      in result
        assert "carcinoma" in result

        # Noise should be gone
        assert not any(ch.isdigit() for ch in result)
        assert "is"  not in result.split()
        assert "a"   not in result.split()
        assert "the" not in result.split()
