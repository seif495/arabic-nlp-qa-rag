from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MS1_DATASET_PATH = Path("data/processed/ms1/ms1_dataset_processed_v001.jsonl")


@dataclass(frozen=True)
class MS2DatasetRecord:
    sample_id: str
    transcript_id: str
    qa_id: str
    normalized_context: str
    question_text: str
    answer_text: str
    split: str


def load_ms1_processed_records(
    path: Path | str = DEFAULT_MS1_DATASET_PATH,
) -> list[MS2DatasetRecord]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        return []

    records: list[MS2DatasetRecord] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {dataset_path}:{line_number}: {exc}"
                ) from exc
            records.append(_record_from_mapping(payload, line_number=line_number))
    return records


def _record_from_mapping(payload: dict[str, Any], line_number: int) -> MS2DatasetRecord:
    return MS2DatasetRecord(
        sample_id=str(payload.get("sample_id") or f"sample_{line_number:06d}"),
        transcript_id=str(payload.get("transcript_id") or ""),
        qa_id=str(payload.get("qa_id") or f"qa_{line_number:06d}"),
        normalized_context=str(payload.get("normalized_context") or ""),
        question_text=str(payload.get("question_text") or ""),
        answer_text=str(payload.get("answer_text") or ""),
        split=str(payload.get("split") or "train"),
    )
