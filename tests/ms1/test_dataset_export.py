from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.ms1.dataset_export import (
    build_processed_dataset_records,
    export_processed_dataset,
    run_dataset_sanity_check,
)


class TestDatasetExport(unittest.TestCase):
    def test_processed_records_preserve_linkage(self) -> None:
        records = build_processed_dataset_records()
        self.assertEqual(len(records), 1)

        record = records[0]
        self.assertEqual(record.transcript_id, "tr_0001")
        self.assertEqual(record.qa_id, "qa_0001")
        self.assertEqual(record.cleaned_record_id, "cln_0001")

    def test_export_writes_dataset_and_summary_in_processed_ms1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            dataset_path = export_processed_dataset(repo_root=repo_root)

            self.assertEqual(
                dataset_path.parent.resolve(),
                (repo_root / "data" / "processed" / "ms1").resolve(),
            )
            self.assertTrue(dataset_path.exists())

            with dataset_path.open(encoding="utf-8") as file_obj:
                exported_record = json.loads(file_obj.readline())
            self.assertEqual(exported_record["qa_id"], "qa_0001")
            self.assertEqual(exported_record["transcript_id"], "tr_0001")

            summary_path = (
                repo_root
                / "data"
                / "processed"
                / "ms1"
                / "ms1_dataset_validation_summary_v001.json"
            )
            self.assertTrue(summary_path.exists())
            with summary_path.open(encoding="utf-8") as file_obj:
                summary = json.load(file_obj)
            self.assertTrue(summary["usable"])
            self.assertTrue(summary["linkage_ok"])

    def test_sanity_check_reports_usable_dataset(self) -> None:
        summary = run_dataset_sanity_check(build_processed_dataset_records())
        self.assertEqual(summary["record_count"], 1)
        self.assertEqual(summary["split_count"], 1)
        self.assertTrue(summary["usable"])


if __name__ == "__main__":
    unittest.main()
