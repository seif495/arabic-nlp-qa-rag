from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ms2.data.length_caps import (
    LENGTH_CAPS,
    analyze_length_distributions,
    run_length_analysis,
)
from src.ms2.data.records import MS2DatasetRecord
from src.ms2.schemas import RunConfig


def _record(context: str = "اهلا بكم في الحلقة") -> MS2DatasetRecord:
    return MS2DatasetRecord(
        sample_id="s1",
        transcript_id="t1",
        qa_id="q1",
        normalized_context=context,
        question_text="ما الموضوع؟",
        answer_text="الحلقة",
        split="train",
    )


def test_length_caps_are_linked_from_run_config() -> None:
    config = RunConfig(model_id="A", seed=13)

    assert config.l_q == LENGTH_CAPS["l_q"]
    assert config.l_c == LENGTH_CAPS["l_c"]
    assert config.l_enc == LENGTH_CAPS["l_enc"]
    assert config.l_dec == LENGTH_CAPS["l_dec"]


def test_length_distribution_reports_percentiles_and_encoder_coverage() -> None:
    records = [_record(), _record(context=" ".join(f"tok{i}" for i in range(20)))]

    stats = analyze_length_distributions(records)

    assert set(stats) == {"question", "context", "encoder", "decoder"}
    assert stats["encoder"].p95 >= stats["question"].p95
    assert stats["encoder"].coverage >= 0.99


def test_run_length_analysis_writes_json_and_histograms(tmp_path: Path) -> None:
    dataset = tmp_path / "data" / "processed" / "ms1" / "ms1_dataset_processed_v001.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "transcript_id": "t1",
                "qa_id": "q1",
                "normalized_context": "اهلا بكم في الحلقة",
                "question_text": "ما الموضوع؟",
                "answer_text": "الحلقة",
                "split": "train",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_length_analysis(repo_root=tmp_path, tokenizer=lambda text: text.split())

    assert result["encoder_coverage_verified"] is True
    assert (tmp_path / "experiments" / "ms2" / "ms2_length_distribution_v001.json").is_file()
    histograms = list((tmp_path / "docs" / "fs" / "artifacts" / "ms2").glob("length_*.png"))
    assert len(histograms) == 8


def test_run_length_analysis_requires_post_tokenizer_callable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a tokenizer callable"):
        run_length_analysis(repo_root=tmp_path)
