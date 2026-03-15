"""Tests for MS1-ANALYSIS-03: Spelling Inconsistency Detection."""

from __future__ import annotations


from src.ms1.spelling.detector import (
    apply_unification,
    detect_spelling_inconsistencies,
)


class TestDetectSpellingInconsistencies:
    def test_detects_alef_variant_group(self) -> None:
        # أحمد and احمد share canonical form احمد
        tokens = ["أحمد"] * 95 + ["احمد"] * 5
        result = detect_spelling_inconsistencies(tokens)
        assert result["stats"]["canonical_groups_with_variants"] >= 1

    def test_high_confidence_group_is_unified(self) -> None:
        # أحمد appears 95 times, احمد 5 times → unified
        tokens = ["أحمد"] * 95 + ["احمد"] * 5
        result = detect_spelling_inconsistencies(tokens)
        assert result["stats"]["unified_groups"] >= 1

    def test_ambiguous_group_not_unified(self) -> None:
        # 60/40 split → below 90 % threshold → ambiguous
        tokens = ["أحمد"] * 60 + ["احمد"] * 40
        result = detect_spelling_inconsistencies(tokens)
        assert result["stats"]["ambiguous_groups"] >= 1
        assert result["stats"]["unified_groups"] == 0

    def test_no_inconsistency_in_uniform_corpus(self) -> None:
        tokens = ["كتاب"] * 100
        result = detect_spelling_inconsistencies(tokens)
        assert result["stats"]["canonical_groups_with_variants"] == 0
        assert result["unified"] == {}

    def test_unified_map_contains_minority_to_dominant(self) -> None:
        tokens = ["أحمد"] * 95 + ["احمد"] * 5
        result = detect_spelling_inconsistencies(tokens)
        # احمد (5) should unify to أحمد (95)
        assert "احمد" in result["unified"]
        assert result["unified"]["احمد"] == "أحمد"

    def test_custom_threshold_respected(self) -> None:
        # 70 % dominance – fails at 90 % but passes at 60 %
        tokens = ["أحمد"] * 70 + ["احمد"] * 30
        result_strict = detect_spelling_inconsistencies(tokens, threshold=0.90)
        result_loose = detect_spelling_inconsistencies(tokens, threshold=0.60)
        assert result_strict["stats"]["ambiguous_groups"] >= 1
        assert result_loose["stats"]["unified_groups"] >= 1

    def test_empty_token_list(self) -> None:
        result = detect_spelling_inconsistencies([])
        assert result["stats"]["canonical_groups_with_variants"] == 0

    def test_alef_maqsura_variant_detected(self) -> None:
        # مصطفى and مصطفي should be treated as variants
        tokens = ["مصطفى"] * 80 + ["مصطفي"] * 20
        result = detect_spelling_inconsistencies(tokens)
        assert result["stats"]["canonical_groups_with_variants"] >= 1


class TestApplyUnification:
    def test_replaces_minority_form(self) -> None:
        unified_map = {"احمد": "أحمد"}
        text = "قال احمد مرحبا"
        result = apply_unification(text, unified_map)
        assert "أحمد" in result
        assert "احمد" not in result

    def test_no_change_when_map_empty(self) -> None:
        text = "نص عربي"
        assert apply_unification(text, {}) == text

    def test_does_not_modify_already_dominant_form(self) -> None:
        unified_map = {"احمد": "أحمد"}
        text = "قال أحمد مرحبا"
        assert apply_unification(text, unified_map) == text
