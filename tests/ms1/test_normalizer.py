"""Tests for MS1-PROC-02: Arabic Character Normalization."""

from __future__ import annotations


from src.ms1.normalization.normalizer import normalize_arabic


class TestNormalizeArabic:
    # --- Tatweel removal ---
    def test_removes_tatweel(self) -> None:
        assert normalize_arabic("كتـاب") == "كتاب"

    def test_removes_multiple_tatweels(self) -> None:
        assert normalize_arabic("كتـــاب") == "كتاب"

    # --- Diacritic removal ---
    def test_removes_fatha(self) -> None:
        assert normalize_arabic("كَتَبَ") == "كتب"

    def test_removes_damma(self) -> None:
        assert normalize_arabic("كُتُب") == "كتب"

    def test_removes_kasra(self) -> None:
        assert normalize_arabic("كِتَاب") == "كتاب"

    def test_removes_shadda(self) -> None:
        assert normalize_arabic("مُحَمَّد") == "محمد"

    def test_removes_sukun(self) -> None:
        assert normalize_arabic("قَلْب") == "قلب"

    # --- Alef normalisation ---
    def test_hamza_above_alef(self) -> None:
        assert normalize_arabic("أحمد") == "احمد"

    def test_hamza_below_alef(self) -> None:
        assert normalize_arabic("إبراهيم") == "ابراهيم"

    def test_madda_above_alef(self) -> None:
        assert normalize_arabic("آخر") == "اخر"

    def test_alef_wasla(self) -> None:
        assert normalize_arabic("ٱلله") == "الله"

    # --- Ya / alef maqsura normalisation ---
    def test_alef_maqsura_to_ya(self) -> None:
        assert normalize_arabic("مصطفى") == "مصطفي"

    def test_alef_maqsura_mid_word(self) -> None:
        # ى within a word should also be normalised
        assert normalize_arabic("رأى") == "راي"

    # --- Flag controls ---
    def test_disable_diacritics(self) -> None:
        result = normalize_arabic("كَتَبَ", remove_diacritics=False)
        assert "َ" in result

    def test_disable_alef_norm(self) -> None:
        result = normalize_arabic("أحمد", normalize_alef=False)
        assert result.startswith("أ")

    def test_disable_ya_norm(self) -> None:
        result = normalize_arabic("مصطفى", normalize_ya=False)
        assert result.endswith("ى")

    def test_disable_tatweel(self) -> None:
        result = normalize_arabic("كتـاب", remove_tatweel=False)
        assert "ـ" in result

    # --- Edge cases ---
    def test_empty_string(self) -> None:
        assert normalize_arabic("") == ""

    def test_latin_chars_unchanged(self) -> None:
        result = normalize_arabic("Hello مرحبا")
        assert "Hello" in result

    def test_idempotent(self) -> None:
        text = "أحمد إبراهيم آخر"
        once = normalize_arabic(text)
        twice = normalize_arabic(once)
        assert once == twice
