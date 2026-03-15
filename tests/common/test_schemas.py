from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.common.schemas import (
    EXAMPLE_CLEANED_RECORD,
    EXAMPLE_PREPARED_DATASET_SAMPLE,
    EXAMPLE_PROCESSED_DATASET_RECORD,
    EXAMPLE_QA_RECORD,
    EXAMPLE_TRANSCRIPT_RECORD,
    CleanedRecord,
    PreparedDatasetSample,
    ProcessedDatasetRecord,
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
        start = EXAMPLE_QA_RECORD.answer_start_char
        end = start + len(EXAMPLE_QA_RECORD.answer_text)
        self.assertEqual(
            EXAMPLE_TRANSCRIPT_RECORD.raw_text[start:end],
            EXAMPLE_QA_RECORD.answer_text,
        )

    def test_cleaned_record_example_matches_schema(self) -> None:
        self.assertIsInstance(EXAMPLE_CLEANED_RECORD, CleanedRecord)
        self.assertEqual(EXAMPLE_CLEANED_RECORD.transcript_id, "tr_0001")
        self.assertTrue(EXAMPLE_CLEANED_RECORD.normalized_text)

    def test_prepared_dataset_example_matches_schema(self) -> None:
        self.assertIsInstance(EXAMPLE_PREPARED_DATASET_SAMPLE, PreparedDatasetSample)
        self.assertEqual(EXAMPLE_PREPARED_DATASET_SAMPLE.qa_id, "qa_0001")
        self.assertIn(EXAMPLE_PREPARED_DATASET_SAMPLE.split, {"train", "val", "test"})

    def test_processed_dataset_example_matches_schema(self) -> None:
        self.assertIsInstance(EXAMPLE_PROCESSED_DATASET_RECORD, ProcessedDatasetRecord)
        self.assertEqual(EXAMPLE_PROCESSED_DATASET_RECORD.transcript_id, "tr_0001")
        self.assertEqual(EXAMPLE_PROCESSED_DATASET_RECORD.qa_id, "qa_0001")
        self.assertEqual(EXAMPLE_PROCESSED_DATASET_RECORD.cleaned_record_id, "cln_0001")
        self.assertIn(EXAMPLE_PROCESSED_DATASET_RECORD.split, {"train", "val", "test"})

    def test_transcript_json_example_is_in_sync(self) -> None:
        transcript_example_path = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "fs"
            / "artifacts"
            / "ms1"
            / "ms1-transcript-record.example.json"
        )
        with transcript_example_path.open(encoding="utf-8") as file_obj:
            transcript_json = json.load(file_obj)
        self.assertEqual(
            transcript_json["transcript_id"], EXAMPLE_TRANSCRIPT_RECORD.transcript_id
        )
        self.assertEqual(
            transcript_json["raw_text"], EXAMPLE_TRANSCRIPT_RECORD.raw_text
        )

    def test_prepared_sample_json_example_is_in_sync(self) -> None:
        prepared_example_path = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "fs"
            / "artifacts"
            / "ms1"
            / "ms1-prepared-dataset-sample.example.json"
        )
        with prepared_example_path.open(encoding="utf-8") as file_obj:
            prepared_json = json.load(file_obj)
        self.assertEqual(
            prepared_json["input_text"], EXAMPLE_PREPARED_DATASET_SAMPLE.input_text
        )

    def test_processed_dataset_jsonl_example_is_in_sync(self) -> None:
        processed_example_path = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "fs"
            / "artifacts"
            / "ms1"
            / "ms1-processed-dataset.sample.jsonl"
        )
        with processed_example_path.open(encoding="utf-8") as file_obj:
            processed_json = json.loads(file_obj.readline())
        self.assertEqual(
            processed_json["sample_id"], EXAMPLE_PROCESSED_DATASET_RECORD.sample_id
        )
        self.assertEqual(
            processed_json["normalized_context"],
            EXAMPLE_PROCESSED_DATASET_RECORD.normalized_context,
        )


if __name__ == "__main__":
    unittest.main()
