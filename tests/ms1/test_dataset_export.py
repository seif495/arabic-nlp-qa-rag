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
    def _create_external_inputs(self, repo_root: Path) -> None:
        transcripts_dir = repo_root / "data" / "external" / "transcripts"
        qa_dir = repo_root / "data" / "external" / "qa"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        qa_dir.mkdir(parents=True, exist_ok=True)

        transcript_path = transcripts_dir / "حلقة تجريبية  الدحيح.txt"
        transcript_path.write_text(
            "0.10: أهلا بكم\n1.20: في حلقة تجريبية\n",
            encoding="utf-8",
        )

        qa_path = qa_dir / "vid_001_QA.csv"
        qa_path.write_text(
            "video_id,video_title,question_id,question,answer,difficulty\n"
            "vid_001,حلقة تجريبية | الدحيح,vid_001_Q001,ما موضوع الحلقة؟,حلقة تجريبية,Easy\n",
            encoding="utf-8",
        )

    def test_processed_records_preserve_linkage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            self._create_external_inputs(repo_root)
            records = build_processed_dataset_records(repo_root=repo_root)

        self.assertEqual(len(records), 1)

        record = records[0]
        self.assertEqual(record.transcript_id, "vid_001")
        self.assertEqual(record.qa_id, "vid_001_Q001")
        self.assertEqual(record.cleaned_record_id, "cln_vid_001_Q001")
        self.assertEqual(record.normalized_context, "أهلا بكم في حلقة تجريبية")

    def test_export_writes_dataset_and_summary_in_processed_ms1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            self._create_external_inputs(repo_root)
            dataset_path = export_processed_dataset(repo_root=repo_root)

            self.assertEqual(
                dataset_path.parent.resolve(),
                (repo_root / "data" / "processed" / "ms1").resolve(),
            )
            self.assertTrue(dataset_path.exists())

            with dataset_path.open(encoding="utf-8") as file_obj:
                exported_record = json.loads(file_obj.readline())
            self.assertEqual(exported_record["qa_id"], "vid_001_Q001")
            self.assertEqual(exported_record["transcript_id"], "vid_001")

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
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            self._create_external_inputs(repo_root)
            summary = run_dataset_sanity_check(
                build_processed_dataset_records(repo_root=repo_root)
            )

        self.assertEqual(summary["record_count"], 1)
        self.assertEqual(summary["split_count"], 1)
        self.assertTrue(summary["usable"])


if __name__ == "__main__":
    unittest.main()
