"""Tests for MS1-PROC-01: Deterministic Cleaning Pipeline."""
from __future__ import annotations

import pytest

from src.ms1.cleaning.cleaner import clean_qa_text, clean_transcript


class TestCleanTranscript:
    def test_removes_integer_timestamp_prefix(self) -> None:
        raw = "12: مرحباً بكم"
        assert clean_transcript(raw) == "مرحباً بكم"

    def test_removes_float_timestamp_prefix(self) -> None:
        raw = "3.076: نص عربي"
        assert clean_transcript(raw) == "نص عربي"

    def test_removes_zero_timestamp(self) -> None:
        raw = "0.0: أول جملة"
        assert clean_transcript(raw) == "أول جملة"

    def test_joins_multiple_lines(self) -> None:
        raw = "0.0: أول جملة\n3.0: ثاني جملة"
        assert clean_transcript(raw) == "أول جملة ثاني جملة"

    def test_removes_empty_lines(self) -> None:
        raw = "0.0: نص\n\n5.0: نص آخر"
        assert clean_transcript(raw) == "نص نص آخر"

    def test_normalises_multiple_spaces(self) -> None:
        raw = "0.0: كلمة   كلمة"
        assert clean_transcript(raw) == "كلمة كلمة"

    def test_strips_leading_trailing_whitespace_per_line(self) -> None:
        raw = "  1.0:   نص   "
        assert clean_transcript(raw) == "نص"

    def test_empty_input(self) -> None:
        assert clean_transcript("") == ""

    def test_no_timestamps_unchanged(self) -> None:
        raw = "نص بدون طوابع زمنية"
        assert clean_transcript(raw) == raw

    def test_deterministic(self) -> None:
        raw = "0.0: جملة\n1.0: جملة أخرى"
        assert clean_transcript(raw) == clean_transcript(raw)

    def test_does_not_remove_non_timestamp_numbers(self) -> None:
        # A number mid-line should not be stripped
        raw = "0.0: لديه 3 أقسام"
        result = clean_transcript(raw)
        assert "3" in result


class TestCleanQaText:
    def test_strips_whitespace(self) -> None:
        assert clean_qa_text("  نص  ") == "نص"

    def test_normalises_multiple_spaces(self) -> None:
        assert clean_qa_text("نص   نص") == "نص نص"

    def test_empty_string(self) -> None:
        assert clean_qa_text("") == ""

    def test_no_change_when_already_clean(self) -> None:
        text = "ما هو الموضوع؟"
        assert clean_qa_text(text) == text
