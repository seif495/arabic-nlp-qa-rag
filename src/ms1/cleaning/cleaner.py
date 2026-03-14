"""
MS1-PROC-01: Deterministic Cleaning Pipeline

Implements conservative deterministic cleaning rules for transcript and QA text.
All rules are order-dependent, deterministic, and repeatable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.common.paths import MS1Paths, make_ms1_output_filename, resolve_ms1_paths
from src.ms1.ingest.loader import load_qa_files, load_transcripts

# Rule 1: timestamp prefixes like "0.0:", "3.076:", "12:"
TIMESTAMP_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)?:\s*")
# Rule 2: multiple consecutive spaces/tabs
MULTI_WHITESPACE_RE = re.compile(r"[ \t]+")


# ---------------------------------------------------------------------------
# Core cleaning functions
# ---------------------------------------------------------------------------


def clean_transcript(raw_text: str) -> str:
    """Apply all cleaning rules to a single transcript string.

    Rules (applied in order):
    1. Timestamp prefix removal  – strips lines starting with ``<number>:``.
    2. Internal whitespace normalisation – collapses runs of spaces/tabs.
    3. Strip per-line leading/trailing whitespace.
    4. Empty-line removal – drops lines that are blank after rules 1-3.
    5. Line joining – merges remaining lines with a single space, reflecting
       that transcripts represent continuous speech.
    """
    cleaned_lines: list[str] = []
    for raw_line in raw_text.splitlines():
        line = TIMESTAMP_PREFIX_RE.sub("", raw_line.strip())
        line = MULTI_WHITESPACE_RE.sub(" ", line).strip()
        if line:
            cleaned_lines.append(line)
    return " ".join(cleaned_lines)


def clean_qa_text(text: str) -> str:
    """Apply deterministic cleaning to a QA question or answer string.

    Rules:
    1. Internal whitespace normalisation.
    2. Strip leading/trailing whitespace.
    """
    return MULTI_WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_cleaning_pipeline(
    paths: MS1Paths | None = None,
    repo_root: Path | None = None,
) -> dict:
    """Run the full deterministic cleaning pipeline and save artifacts.

    Loads all transcripts and QA records, applies cleaning rules, then writes
    three artifacts to ``experiments/ms1/``:

    * ``ms1_cleaning_rules_v001.md``     – human-readable rule documentation
    * ``ms1_cleaning_before_after_v001.json`` – before/after text examples
    * ``ms1_cleaning_summary_v001.json`` – per-transcript stats + totals

    Returns the summary dict.
    """
    resolved_paths = paths or resolve_ms1_paths(repo_root=repo_root)
    transcripts = load_transcripts(resolved_paths)
    qa_records = load_qa_files(resolved_paths)

    transcript_stats: list[dict] = []
    before_after_transcript: list[dict] = []

    for transcript in transcripts:
        raw_text: str = transcript["content"]
        cleaned_text = clean_transcript(raw_text)

        raw_len = len(raw_text)
        cleaned_len = len(cleaned_text)
        transcript_stats.append(
            {
                "transcript_id": transcript["video_id"],
                "raw_char_count": raw_len,
                "cleaned_char_count": cleaned_len,
                "chars_removed": raw_len - cleaned_len,
                "reduction_pct": round(
                    100.0 * (raw_len - cleaned_len) / max(raw_len, 1), 2
                ),
            }
        )

        # Keep up to 5 before/after examples
        if len(before_after_transcript) < 5:
            before_after_transcript.append(
                {
                    "transcript_id": transcript["video_id"],
                    "before": raw_text[:400],
                    "after": cleaned_text[:400],
                }
            )

    # QA cleaning stats
    qa_chars_before = 0
    qa_chars_after = 0
    before_after_qa: list[dict] = []

    for record in qa_records:
        q_raw = record.get("question", "")
        a_raw = record.get("answer", "")
        q_clean = clean_qa_text(q_raw)
        a_clean = clean_qa_text(a_raw)
        qa_chars_before += len(q_raw) + len(a_raw)
        qa_chars_after += len(q_clean) + len(a_clean)

        if len(before_after_qa) < 5:
            before_after_qa.append(
                {
                    "question_before": q_raw,
                    "question_after": q_clean,
                    "answer_before": a_raw,
                    "answer_after": a_clean,
                }
            )

    summary: dict = {
        "transcripts_cleaned": len(transcripts),
        "qa_records_cleaned": len(qa_records),
        "total_transcript_chars_before": sum(
            s["raw_char_count"] for s in transcript_stats
        ),
        "total_transcript_chars_after": sum(
            s["cleaned_char_count"] for s in transcript_stats
        ),
        "total_transcript_chars_removed": sum(
            s["chars_removed"] for s in transcript_stats
        ),
        "qa_chars_before": qa_chars_before,
        "qa_chars_after": qa_chars_after,
        "qa_chars_removed": qa_chars_before - qa_chars_after,
        "transcript_stats": transcript_stats,
    }

    output_dir = resolved_paths.experiments_ms1

    # Artifact 1: cleaning rules document
    rules_path = output_dir / make_ms1_output_filename("cleaning", "rules", 1, "md")
    _write_rules_doc(rules_path)

    # Artifact 2: before/after examples
    examples_path = output_dir / make_ms1_output_filename(
        "cleaning", "before_after", 1, "json"
    )
    with examples_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "transcript_examples": before_after_transcript,
                "qa_examples": before_after_qa,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    # Artifact 3: cleaning effect summary table
    summary_path = output_dir / make_ms1_output_filename(
        "cleaning", "summary", 1, "json"
    )
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    return summary


# ---------------------------------------------------------------------------
# Artifact helper
# ---------------------------------------------------------------------------


def _write_rules_doc(path: Path) -> None:
    content = """\
# MS1 Cleaning Rules Document

## Overview

Deterministic rules applied to all transcript and QA text in Milestone 1.
Rules are applied sequentially; every run on the same input produces the same output.
The originals in `data/external/` are never modified.

---

## Rules Applied to Transcripts

### Rule 1 – Timestamp Prefix Removal

| Property | Value |
|---|---|
| Pattern | `^\\d+(?:\\.\\d+)?:\\s*` |
| Action | Remove matched prefix from each line |
| Rationale | YouTube-style timestamps (`0.0:`, `12.345:`) are structural artefacts of the transcript export format, not part of the spoken content. |

### Rule 2 – Internal Whitespace Normalisation

| Property | Value |
|---|---|
| Pattern | One or more consecutive spaces or tabs |
| Action | Replace with a single ASCII space |
| Rationale | Prevents inconsistent tokenisation downstream. |

### Rule 3 – Per-Line Strip

| Property | Value |
|---|---|
| Action | `str.strip()` on each line after rules 1–2 |
| Rationale | Removes residual leading/trailing whitespace. |

### Rule 4 – Empty-Line Removal

| Property | Value |
|---|---|
| Condition | Line is empty or whitespace-only after rules 1–3 |
| Action | Drop the line |
| Rationale | Blank lines carry no semantic content in a speech transcript. |

### Rule 5 – Line Joining

| Property | Value |
|---|---|
| Action | Join remaining lines with a single space |
| Rationale | Transcripts represent continuous speech; line breaks reflect the export format, not sentence boundaries. |

---

## Rules Applied to QA Text

### Rule 1 – Internal Whitespace Normalisation
Same as transcript Rule 2.

### Rule 2 – Strip
Same as transcript Rule 3.

---

## Non-Destructive Guarantees

- All rules are **deterministic** and **repeatable**.
- No semantic Arabic content is altered; only formatting artefacts are removed.
- Original files in `data/external/` are **never** modified.
- Cleaned text is kept in memory / interim paths; raw data remains intact.
"""
    path.write_text(content, encoding="utf-8")
