from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.ms2 import orchestration


def test_run_all_executes_full_matrix_and_writes_pipeline_artifacts(
    tmp_path: Path,
) -> None:
    results = list(orchestration.run_all(repo_root=tmp_path))

    counts = Counter(result.command for result in results)
    assert counts == {
        "analyze-lengths": 1,
        "prep-data": 2,
        "train": 6,
        "infer": 6,
        "evaluate": 6,
        "evaluate-protocol": 1,
        "ablate": 3,
        "compare": 1,
    }
    assert all(result.status == "ok" for result in results)
    assert all(result.elapsed_seconds is not None for result in results)

    for result in results:
        assert Path(result.output_path).exists()

    experiments_dir = tmp_path / "experiments" / "ms2"
    assert len(list(experiments_dir.glob("run_all_log_v*.txt"))) == 1
    assert (experiments_dir / "ms2_pipeline_integration_checklist_v001.md").is_file()
    assert (experiments_dir / "ms2_pipeline_known_limitations_v001.md").is_file()


def test_run_all_is_idempotent_when_outputs_exist(tmp_path: Path) -> None:
    list(orchestration.run_all(repo_root=tmp_path))
    rerun = list(orchestration.run_all(repo_root=tmp_path))

    assert rerun
    assert all(result.status == "skipped" for result in rerun)
    experiments_dir = tmp_path / "experiments" / "ms2"
    assert len(list(experiments_dir.glob("run_all_log_v*.txt"))) == 1


def test_run_all_force_from_reexecutes_selected_stage_onward(tmp_path: Path) -> None:
    list(orchestration.run_all(repo_root=tmp_path))
    forced = list(orchestration.run_all(repo_root=tmp_path, force_from="train"))

    first_train_index = next(
        index for index, result in enumerate(forced) if result.command == "train"
    )
    assert all(result.status == "skipped" for result in forced[:first_train_index])
    assert all(result.status == "ok" for result in forced[first_train_index:])


def test_run_all_emits_training_budget_warnings_when_budget_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orchestration, "TRAINING_BUDGET_MINUTES", 0)
    results = list(orchestration.run_all(repo_root=tmp_path))

    train_results = [result for result in results if result.command == "train"]
    assert train_results
    assert all(result.budget_warning for result in train_results)
