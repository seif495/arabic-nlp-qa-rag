from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.ms1.orchestration import run_all


class TestMS1PipelineRunAll(unittest.TestCase):
    def test_run_all_executes_pipeline_and_generates_integration_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            self._seed_minimal_external_dataset(repo_root)

            results = run_all(repo_root=repo_root)

            self.assertEqual(
                [result.command for result in results],
                ["profile", "detect-irregularities", "normalize", "build-dataset"],
            )

            experiments_dir = repo_root / "experiments" / "ms1"
            processed_dir = repo_root / "data" / "processed" / "ms1"
            normalized_dir = repo_root / "data" / "interim" / "normalized"

            self.assertTrue(experiments_dir.exists())
            self.assertTrue(processed_dir.exists())
            self.assertTrue(normalized_dir.exists())

            execution_log = experiments_dir / "ms1_pipeline_execution_log_v001.md"
            artifact_manifest = (
                experiments_dir / "ms1_pipeline_artifact_manifest_v001.json"
            )
            integration_checklist = (
                experiments_dir / "ms1_pipeline_integration_checklist_v001.md"
            )
            known_limitations = (
                experiments_dir / "ms1_pipeline_known_limitations_v001.md"
            )

            self.assertTrue(execution_log.exists())
            self.assertTrue(artifact_manifest.exists())
            self.assertTrue(integration_checklist.exists())
            self.assertTrue(known_limitations.exists())

            log_text = execution_log.read_text(encoding="utf-8")
            self.assertIn("`profile` status=ok", log_text)
            self.assertIn("`detect-irregularities` status=ok", log_text)
            self.assertIn("`normalize` status=ok", log_text)
            self.assertIn("`build-dataset` status=ok", log_text)

            manifest_payload = json.loads(artifact_manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest_payload["commands_executed"],
                ["profile", "detect-irregularities", "normalize", "build-dataset"],
            )
            self.assertEqual(
                manifest_payload["canonical_directories"]["data_processed_ms1"],
                "data/processed/ms1",
            )
            self.assertEqual(
                manifest_payload["canonical_directories"]["experiments_ms1"],
                "experiments/ms1",
            )
            self.assertEqual(
                manifest_payload["command_outputs"]["profile"],
                "experiments/ms1",
            )
            self.assertEqual(
                manifest_payload["command_outputs"]["detect-irregularities"],
                "experiments/ms1",
            )
            self.assertEqual(
                manifest_payload["command_outputs"]["normalize"],
                "data/interim/normalized",
            )
            self.assertEqual(
                manifest_payload["command_outputs"]["build-dataset"],
                "data/processed/ms1",
            )
            for output_path in manifest_payload["command_outputs"].values():
                self.assertNotIn("\\", output_path)
            self.assertNotIn(
                str(repo_root),
                log_text,
            )

    def _seed_minimal_external_dataset(self, repo_root: Path) -> None:
        transcripts_dir = repo_root / "data" / "external" / "transcripts"
        qa_dir = repo_root / "data" / "external" / "qa"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        qa_dir.mkdir(parents=True, exist_ok=True)

        (transcripts_dir / "video1.txt").write_text(
            "0.0: أهلا وسهلا\n1.0: مرحبا بكم", encoding="utf-8"
        )
        (qa_dir / "video1_QA.csv").write_text(
            "video_id,question_id,video_title,question,answer\n"
            "video1,video1_Q1,video1,ما الموضوع؟,الموضوع مقدمة\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
