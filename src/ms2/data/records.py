from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.common.paths import MS2Paths


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


def load_ms2_cleaned_input_records(paths: "MS2Paths") -> list[MS2DatasetRecord]:
    from src.common.paths import list_ms2_cleaned_input_files

    files = list_ms2_cleaned_input_files(paths)
    if not files:
        raise ValueError(
            f"No *.json files found under {paths.data_external_ms2_cleaned_input}"
        )

    records: list[MS2DatasetRecord] = []
    seen_ids: set[str] = set()

    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if "data" not in payload:
            raise ValueError(f"{file_path}: missing 'data' key")

        for data_idx, data_item in enumerate(payload["data"]):
            topic = str(data_item.get("title") or file_path.stem)
            paragraphs = data_item.get("paragraphs")
            if not paragraphs:
                raise ValueError(f"{file_path} data[{data_idx}]: missing or empty 'paragraphs'")

            for para_idx, paragraph in enumerate(paragraphs):
                context = paragraph.get("context", "")
                if not context:
                    raise ValueError(
                        f"{file_path} data[{data_idx}].paragraphs[{para_idx}]: empty 'context'"
                    )
                qas = paragraph.get("qas")
                if not qas:
                    raise ValueError(
                        f"{file_path} data[{data_idx}].paragraphs[{para_idx}]: missing or empty 'qas'"
                    )

                for qa_idx, qa in enumerate(qas):
                    qa_id = str(qa.get("id") or f"qa_{data_idx}_{para_idx}_{qa_idx}")
                    question = qa.get("question", "")
                    if not question:
                        raise ValueError(
                            f"{file_path} qas[{qa_idx}] id={qa_id}: empty 'question'"
                        )
                    answers = qa.get("answers")
                    if not answers:
                        raise ValueError(
                            f"{file_path} qas[{qa_idx}] id={qa_id}: no answers"
                        )
                    answer_text = answers[0].get("text", "")
                    if not answer_text:
                        raise ValueError(
                            f"{file_path} qas[{qa_idx}] id={qa_id}: empty answer text"
                        )

                    sample_id = f"{topic}:{qa_id}"
                    if sample_id in seen_ids:
                        raise ValueError(f"duplicate sample_id={sample_id} in {file_path}")
                    seen_ids.add(sample_id)

                    records.append(
                        MS2DatasetRecord(
                            sample_id=sample_id,
                            transcript_id=topic,
                            qa_id=qa_id,
                            normalized_context=context,
                            question_text=question,
                            answer_text=answer_text,
                            split=_assign_split(file_path, topic, qa_id),
                        )
                    )

    return records


def _assign_split(file_path: Path, topic: str, qa_id: str) -> str:
    if file_path.stem.endswith("_test_set"):
        return "test"
    key = f"{topic}:{qa_id}".encode("utf-8")
    h = int(hashlib.md5(key).hexdigest(), 16)
    return "dev" if (h % 100) < 10 else "train"


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
