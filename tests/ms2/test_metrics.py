from __future__ import annotations

import unittest

from src.ms2.metrics.normalize import arabic_post_normalize
from src.ms2.metrics.scoring import (
    aggregate_metrics,
    bleu1,
    char_edit_distance_normalized,
    exact_match,
    token_f1,
)


class TestArabicPostNormalize(unittest.TestCase):
    def test_table_driven_normalization_cases(self) -> None:
        cases = [
            ("أ", "ا"),
            ("إ", "ا"),
            ("آ", "ا"),
            ("ى", "ي"),
            ("أحمد", "احمد"),
            ("إيمان", "ايمان"),
            ("آثار", "اثار"),
            ("فتى", "فتي"),
            ("مُحَمَّد", "محمد"),
            ("كِتاب", "كتاب"),
            ("سؤالٌ", "سؤال"),
            ("جوابٍ", "جواب"),
            ("قصةً", "قصة"),
            ("ٱختبار", "ٱختبار"),
            ("اٰ", "ا"),
            ("نصّ", "نص"),
            ("مـــا", "مـــا"),
            ("  سؤال   وجواب ", "سؤال وجواب"),
            ("أَسْئِلَة", "اسئلة"),
            ("إِلَى", "الي"),
            ("آلَى", "الي"),
            ("الذَّهَب الأصْفَر", "الذهب الاصفر"),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(arabic_post_normalize(source), expected)


class TestScoringMetrics(unittest.TestCase):
    def test_exact_match_uses_normalization(self) -> None:
        self.assertEqual(exact_match("الذَّهَب الأصْفَر", "الذهب الاصفر"), 1.0)
        self.assertEqual(exact_match("الذهب", "الفضة"), 0.0)

    def test_token_f1_cases(self) -> None:
        self.assertEqual(token_f1("الذهب الأصفر", "الذهب الاصفر"), 1.0)
        self.assertEqual(token_f1("الذهب", "الفضة"), 0.0)
        self.assertAlmostEqual(token_f1("الذهب الاصفر", "الذهب الابيض"), 0.5)
        self.assertAlmostEqual(token_f1("الذهب", "الذهب الاصفر"), 2 * 1.0 * 0.5 / 1.5)
        self.assertAlmostEqual(token_f1("الذهب الاصفر اللامع", "الذهب الاصفر"), 2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0))

    def test_char_edit_distance_cases(self) -> None:
        self.assertEqual(char_edit_distance_normalized("أحمد", "احمد"), 0.0)
        self.assertEqual(char_edit_distance_normalized("", ""), 0.0)
        self.assertEqual(char_edit_distance_normalized("", "abc"), 1.0)
        self.assertAlmostEqual(char_edit_distance_normalized("abc", "adc"), 1 / 3)

    def test_bleu1_cases(self) -> None:
        self.assertEqual(bleu1("الذهب الاصفر", "الذهب الاصفر"), 1.0)
        self.assertEqual(bleu1("الذهب", "الفضة"), 0.0)
        self.assertAlmostEqual(bleu1("الذهب الاصفر اللامع", "الذهب الاصفر"), 2 / 3)
        self.assertLess(bleu1("الذهب", "الذهب الاصفر"), 1.0)

    def test_aggregate_metrics(self) -> None:
        metrics = aggregate_metrics(["أحمد", "الذهب"], ["احمد", "الفضة"])
        self.assertEqual(metrics["em"], 0.5)
        self.assertEqual(metrics["token_f1"], 0.5)
        self.assertIn("char_edit_distance", metrics)
        self.assertIn("bleu1", metrics)

    def test_aggregate_rejects_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_metrics(["a"], [])


if __name__ == "__main__":
    unittest.main()
