from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.common.paths import make_ms1_output_filename, resolve_ms1_paths
from src.common.schemas import (
    EXAMPLE_CLEANED_RECORD,
    EXAMPLE_PROCESSED_DATASET_RECORD,
    EXAMPLE_QA_RECORD,
    ProcessedDatasetRecord,
)


def build_processed_dataset_records() -> list[ProcessedDatasetRecord]:
    if EXAMPLE_QA_RECORD.transcript_id != EXAMPLE_CLEANED_RECORD.transcript_id:
        raise ValueError("QA and cleaned record transcript IDs must match")

    return [EXAMPLE_PROCESSED_DATASET_RECORD]


def run_dataset_sanity_check(
    records: list[ProcessedDatasetRecord],
) -> dict[str, int | bool]:
    linkage_ok = all(record.transcript_id for record in records) and all(
        record.qa_id and record.cleaned_record_id for record in records
    )
    usable = bool(records) and linkage_ok
    split_count = len({record.split for record in records})
    return {
        "record_count": len(records),
        "split_count": split_count,
        "linkage_ok": bool(linkage_ok),
        "usable": usable,
    }


def export_processed_dataset(repo_root: Path | None = None) -> Path:
    paths = resolve_ms1_paths(repo_root=repo_root)
    dataset_file = paths.data_processed_ms1 / make_ms1_output_filename(
        stage="dataset",
        name="processed",
        version=1,
        ext="jsonl",
    )

    records = build_processed_dataset_records()
    with dataset_file.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    summary_file = paths.data_processed_ms1 / make_ms1_output_filename(
        stage="dataset",
        name="validation_summary",
        version=1,
        ext="json",
    )
    summary = run_dataset_sanity_check(records)
    with summary_file.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")

    return dataset_file
