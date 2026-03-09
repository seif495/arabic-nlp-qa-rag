from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict
from pathlib import Path

from src.common.paths import MS1Paths, make_ms1_output_filename, resolve_ms1_paths
from src.common.schemas import ProcessedDatasetRecord


TIMESTAMP_PREFIX_PATTERN = re.compile(r"^\d+(?:\.\d+)?:\s*")


def _canonical_title(value: str) -> str:
    return " ".join(value.replace(".txt", "").replace("|", " ").split())


def _normalize_transcript_text(raw_text: str) -> str:
    cleaned_lines: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = TIMESTAMP_PREFIX_PATTERN.sub("", stripped)
        if stripped:
            cleaned_lines.append(stripped)
    return " ".join(cleaned_lines)


def _load_transcript_contexts(transcripts_dir: Path) -> dict[str, str]:
    transcript_contexts: dict[str, str] = {}
    for transcript_file in sorted(transcripts_dir.glob("*.txt")):
        with transcript_file.open(encoding="utf-8") as file_obj:
            transcript_contexts[_canonical_title(transcript_file.stem)] = (
                _normalize_transcript_text(file_obj.read())
            )
    return transcript_contexts


def _extract_transcript_token(identifier: str) -> str:
    token = identifier.removeprefix("cln_")
    if "_Q" in token:
        return token.split("_Q", maxsplit=1)[0]
    return token


def _record_linkage_is_consistent(record: ProcessedDatasetRecord) -> bool:
    if not (record.transcript_id and record.qa_id and record.cleaned_record_id):
        return False

    qa_transcript_token = _extract_transcript_token(record.qa_id)
    cleaned_transcript_token = _extract_transcript_token(record.cleaned_record_id)
    return (
        record.cleaned_record_id.startswith("cln_")
        and qa_transcript_token == record.transcript_id
        and cleaned_transcript_token == record.transcript_id
    )


def build_processed_dataset_records(
    repo_root: Path | None = None,
    paths: MS1Paths | None = None,
) -> list[ProcessedDatasetRecord]:
    resolved_paths = paths or resolve_ms1_paths(repo_root=repo_root)
    transcript_contexts = _load_transcript_contexts(
        resolved_paths.data_external / "transcripts"
    )
    records: list[ProcessedDatasetRecord] = []

    for qa_file in sorted((resolved_paths.data_external / "qa").glob("*_QA.csv")):
        with qa_file.open(encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                transcript_id = row.get("video_id", "").strip()
                qa_id = row.get("question_id", "").strip()
                title_key = _canonical_title(row.get("video_title", ""))
                normalized_context = transcript_contexts.get(title_key, "")
                records.append(
                    ProcessedDatasetRecord(
                        sample_id=f"sample_{qa_id}",
                        transcript_id=transcript_id,
                        qa_id=qa_id,
                        cleaned_record_id=f"cln_{qa_id}",
                        normalized_context=normalized_context,
                        question_text=row.get("question", "").strip(),
                        answer_text=row.get("answer", "").strip(),
                        split="train",
                    )
                )

    return records


def run_dataset_sanity_check(
    records: list[ProcessedDatasetRecord],
) -> dict[str, int | bool]:
    if not records:
        return {
            "record_count": 0,
            "split_count": 0,
            "linkage_ok": False,
            "usable": False,
        }

    linkage_ok = all(_record_linkage_is_consistent(record) for record in records)
    usable = bool(records) and linkage_ok
    split_count = len({record.split for record in records})
    return {
        "record_count": len(records),
        "split_count": split_count,
        "linkage_ok": bool(linkage_ok),
        "usable": usable,
    }


def export_processed_dataset(
    repo_root: Path | None = None,
    paths: MS1Paths | None = None,
) -> Path:
    resolved_paths = paths or resolve_ms1_paths(repo_root=repo_root)
    dataset_file = resolved_paths.data_processed_ms1 / make_ms1_output_filename(
        stage="dataset",
        name="processed",
        version=1,
        ext="jsonl",
    )

    records = build_processed_dataset_records(paths=resolved_paths)
    with dataset_file.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    summary_file = resolved_paths.data_processed_ms1 / make_ms1_output_filename(
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
