from __future__ import annotations

import unittest

from src.common.schemas import (
    EXAMPLE_CLEANED_RECORD,
    EXAMPLE_PREPARED_DATASET_SAMPLE,
    EXAMPLE_QA_RECORD,
    EXAMPLE_TRANSCRIPT_RECORD,
    CleanedRecord,
    PreparedDatasetSample,
    QARecord,
    TranscriptRecord,
)


class TestSharedSchemas(unittest.TestCase):
    def test_transcript_example_matches_schema(self) -> None:
        self.assertIsInstance(EXAMPLE_TRANSCRIPT_RECORD, TranscriptRecord)
        self.assertEqual(EXAMPLE_TRANSCRIPT_RECORD.transcript_id, "tr_0001")
        self.assertEqual(EXAMPLE_TRANSCRIPT_RECORD.language, "ar")

    def test_qa_example_matches_schema(self) -> None:
        self.assertIsInstance(EXAMPLE_QA_RECORD, QARecord)
        self.assertEqual(EXAMPLE_QA_RECORD.transcript_id, "tr_0001")
        self.assertGreaterEqual(EXAMPLE_QA_RECORD.answer_start_char, 0)

    def test_cleaned_record_example_matches_schema(self) -> None:
        self.assertIsInstance(EXAMPLE_CLEANED_RECORD, CleanedRecord)
        self.assertEqual(EXAMPLE_CLEANED_RECORD.transcript_id, "tr_0001")
        self.assertTrue(EXAMPLE_CLEANED_RECORD.normalized_text)

    def test_prepared_dataset_example_matches_schema(self) -> None:
        self.assertIsInstance(EXAMPLE_PREPARED_DATASET_SAMPLE, PreparedDatasetSample)
        self.assertEqual(EXAMPLE_PREPARED_DATASET_SAMPLE.qa_id, "qa_0001")
        self.assertIn(EXAMPLE_PREPARED_DATASET_SAMPLE.split, {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
