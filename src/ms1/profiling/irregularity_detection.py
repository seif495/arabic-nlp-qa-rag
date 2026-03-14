"""Pattern and noise detection for MS1-ANALYSIS-02.

Detects linguistic irregularities and structural noise across transcripts and QA text,
then saves report-ready JSON and Markdown artifacts.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from src.common.paths import MS1Paths, resolve_ms1_paths
from src.ms1.ingest.loader import load_qa_files, load_transcripts

JSON_ARTIFACT_NAME = "irregularity_report.json"
MARKDOWN_ARTIFACT_NAME = "irregularity_summary.md"
MAX_EXAMPLES_PER_CATEGORY = 5
MAX_EXCERPT_CHARS = 180

ARABIC_WORD_RE = re.compile(r"[ء-ي]+")
LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-/]*")
REPEATED_PUNCT_RE = re.compile(r"([!؟?.،,؛;:])\1{1,}")
ELLIPSIS_RE = re.compile(r"\.\.\.|…")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+[،؛؟,;.!?]")
TIMESTAMP_PREFIX_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*:\s*")
BULLET_PREFIX_RE = re.compile(r"^\s*[-•*]\s+")
MULTISPACE_RE = re.compile(r" {2,}")
SYMBOL_CLUTTER_RE = re.compile(r"[`~_=]{2,}|[<>]{2,}")
DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")
TATWEEL_RE = re.compile(r"ـ")
VARIANT_CHAR_RE = re.compile(r"[أإآٱىةؤئ]")
REPEATED_LETTER_RE = re.compile(r"([ء-يA-Za-z])\1{2,}")
REPEATED_WORD_RE = re.compile(r"\b([ء-يA-Za-z]+)\b(?:\W+\1\b){1,}", re.IGNORECASE)

ARABIC_PUNCTUATION = {"،", "؛", "؟"}
LATIN_PUNCTUATION = {",", ";", "?"}
ORTHOGRAPHIC_TRANSLATION_TABLE = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ؤ": "و",
        "ئ": "ي",
        "ى": "ي",
        "ة": "ه",
    }
)


@dataclass(frozen=True)
class TextUnit:
    unit_id: str
    source_type: str
    video_id: str
    source_file: str
    field_name: str
    text: str
    line_number: int | None = None


@dataclass(frozen=True)
class DetectionSignal:
    category: str
    signal: str
    reason: str


@dataclass(frozen=True)
class VariantFamily:
    normalized: str
    forms: tuple[str, ...]
    unit_ids: tuple[str, ...]


Detector = Callable[[TextUnit], list[DetectionSignal]]


def _build_text_units(paths: MS1Paths) -> tuple[list[TextUnit], dict[str, object]]:
    transcripts = load_transcripts(paths)
    qa_records = load_qa_files(paths)

    text_units: list[TextUnit] = []

    for transcript in transcripts:
        video_id = str(transcript.get("video_id", "")).strip()
        source_file = str(transcript.get("filename", "")).strip()
        content = str(transcript.get("content", ""))

        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            line_text = raw_line.strip()
            if not line_text:
                continue

            text_units.append(
                TextUnit(
                    unit_id=f"transcript:{video_id}:{line_number}",
                    source_type="transcript_line",
                    video_id=video_id,
                    source_file=source_file,
                    field_name="content",
                    text=line_text,
                    line_number=line_number,
                )
            )

    for record_index, record in enumerate(qa_records, start=1):
        video_id = str(record.get("source_video_id") or record.get("video_id") or "")
        source_file = str(record.get("source_file") or "")

        question = str(record.get("question") or "").strip()
        if question:
            text_units.append(
                TextUnit(
                    unit_id=f"qa:{record_index}:question",
                    source_type="qa_question",
                    video_id=video_id,
                    source_file=source_file,
                    field_name="question",
                    text=question,
                )
            )

        answer = str(record.get("answer") or "").strip()
        if answer:
            text_units.append(
                TextUnit(
                    unit_id=f"qa:{record_index}:answer",
                    source_type="qa_answer",
                    video_id=video_id,
                    source_file=source_file,
                    field_name="answer",
                    text=answer,
                )
            )

    source_breakdown = dict(Counter(unit.source_type for unit in text_units))
    input_summary: dict[str, object] = {
        "transcript_files": len(transcripts),
        "qa_records": len(qa_records),
        "text_units_total": len(text_units),
        "text_units_by_source": source_breakdown,
    }
    return text_units, input_summary


def _detect_punctuation_inconsistency(unit: TextUnit) -> list[DetectionSignal]:
    text = unit.text
    signals: list[DetectionSignal] = []

    if REPEATED_PUNCT_RE.search(text):
        signals.append(
            DetectionSignal(
                category="punctuation_inconsistency",
                signal="repeated_punctuation",
                reason="Repeated punctuation marks detected.",
            )
        )

    if ELLIPSIS_RE.search(text):
        signals.append(
            DetectionSignal(
                category="punctuation_inconsistency",
                signal="ellipsis_pattern",
                reason="Ellipsis punctuation pattern found.",
            )
        )

    has_arabic_punct = any(char in ARABIC_PUNCTUATION for char in text)
    has_latin_punct = any(char in LATIN_PUNCTUATION for char in text)
    if has_arabic_punct and has_latin_punct:
        signals.append(
            DetectionSignal(
                category="punctuation_inconsistency",
                signal="mixed_punctuation_systems",
                reason="Arabic and Latin punctuation are mixed in the same text unit.",
            )
        )

    if SPACE_BEFORE_PUNCT_RE.search(text):
        signals.append(
            DetectionSignal(
                category="punctuation_inconsistency",
                signal="space_before_punctuation",
                reason="Whitespace detected before punctuation mark.",
            )
        )

    return signals


def _detect_code_switching(unit: TextUnit) -> list[DetectionSignal]:
    tokens = sorted({token for token in LATIN_TOKEN_RE.findall(unit.text)})
    if not tokens:
        return []

    token_preview = ", ".join(tokens[:3])
    return [
        DetectionSignal(
            category="code_switching",
            signal="latin_token_presence",
            reason=f"Latin token(s) detected: {token_preview}",
        )
    ]


def _detect_orthographic_variation(unit: TextUnit) -> list[DetectionSignal]:
    text = unit.text
    signals: list[DetectionSignal] = []

    if DIACRITICS_RE.search(text):
        signals.append(
            DetectionSignal(
                category="orthographic_variation",
                signal="diacritics_presence",
                reason="Arabic diacritics detected.",
            )
        )

    if TATWEEL_RE.search(text):
        signals.append(
            DetectionSignal(
                category="orthographic_variation",
                signal="tatweel_presence",
                reason="Tatweel character detected.",
            )
        )

    alif_forms = {char for char in text if char in {"ا", "أ", "إ", "آ", "ٱ"}}
    if len(alif_forms) >= 2 and any(char != "ا" for char in alif_forms):
        signals.append(
            DetectionSignal(
                category="orthographic_variation",
                signal="mixed_alif_forms",
                reason="Multiple alif letter forms co-occur in the same text unit.",
            )
        )

    if "ى" in text and "ي" in text:
        signals.append(
            DetectionSignal(
                category="orthographic_variation",
                signal="ya_alef_maksura_mix",
                reason="Both ي and ى appear in the same text unit.",
            )
        )

    return signals


def _detect_formatting_artifacts(unit: TextUnit) -> list[DetectionSignal]:
    text = unit.text
    signals: list[DetectionSignal] = []

    if TIMESTAMP_PREFIX_RE.search(text):
        signals.append(
            DetectionSignal(
                category="formatting_artifacts",
                signal="timestamp_prefix",
                reason="Timestamp-like prefix detected.",
            )
        )

    if BULLET_PREFIX_RE.search(text):
        signals.append(
            DetectionSignal(
                category="formatting_artifacts",
                signal="bullet_prefix",
                reason="Bullet-like prefix detected.",
            )
        )

    if MULTISPACE_RE.search(text):
        signals.append(
            DetectionSignal(
                category="formatting_artifacts",
                signal="multi_space",
                reason="Multiple adjacent spaces detected.",
            )
        )

    if SYMBOL_CLUTTER_RE.search(text):
        signals.append(
            DetectionSignal(
                category="formatting_artifacts",
                signal="symbol_clutter",
                reason="Repeated symbol clutter detected.",
            )
        )

    return signals


def _detect_repetition_emphasis(unit: TextUnit) -> list[DetectionSignal]:
    signals: list[DetectionSignal] = []

    if REPEATED_LETTER_RE.search(unit.text):
        signals.append(
            DetectionSignal(
                category="repetition_emphasis",
                signal="repeated_letter_run",
                reason="Character repetition sequence detected.",
            )
        )

    if REPEATED_WORD_RE.search(unit.text):
        signals.append(
            DetectionSignal(
                category="repetition_emphasis",
                signal="repeated_word_sequence",
                reason="Repeated word sequence detected.",
            )
        )

    return signals


DETECTORS: tuple[Detector, ...] = (
    _detect_punctuation_inconsistency,
    _detect_code_switching,
    _detect_orthographic_variation,
    _detect_formatting_artifacts,
    _detect_repetition_emphasis,
)


def _normalize_arabic_token(token: str) -> str:
    normalized = DIACRITICS_RE.sub("", token)
    normalized = normalized.replace("ـ", "")
    normalized = normalized.translate(ORTHOGRAPHIC_TRANSLATION_TABLE)
    return normalized


def _collect_variant_families(text_units: list[TextUnit]) -> list[VariantFamily]:
    normalized_forms: dict[str, set[str]] = defaultdict(set)
    normalized_unit_ids: dict[str, set[str]] = defaultdict(set)

    for unit in text_units:
        for token in ARABIC_WORD_RE.findall(unit.text):
            if len(token) < 3:
                continue

            normalized = _normalize_arabic_token(token)
            if len(normalized) < 3:
                continue

            normalized_forms[normalized].add(token)
            normalized_unit_ids[normalized].add(unit.unit_id)

    families: list[VariantFamily] = []
    for normalized, forms in normalized_forms.items():
        if len(forms) < 2:
            continue
        if len(normalized_unit_ids[normalized]) < 2:
            continue
        if not any(VARIANT_CHAR_RE.search(form) for form in forms):
            continue

        families.append(
            VariantFamily(
                normalized=normalized,
                forms=tuple(sorted(forms)),
                unit_ids=tuple(sorted(normalized_unit_ids[normalized])),
            )
        )

    families.sort(key=lambda item: (-len(item.unit_ids), item.normalized))
    return families


def _format_excerpt(text: str, max_chars: int = MAX_EXCERPT_CHARS) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1] + "…"


def _analyze_irregularities(
    text_units: list[TextUnit], input_summary: dict[str, object]
) -> dict[str, object]:
    unit_index = {unit.unit_id: unit for unit in text_units}

    category_unit_ids: dict[str, set[str]] = defaultdict(set)
    category_signal_counts: dict[str, Counter[str]] = defaultdict(Counter)
    category_examples: dict[str, list[dict[str, object]]] = defaultdict(list)
    category_example_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)

    def register_signal(unit: TextUnit, signal: DetectionSignal) -> None:
        category = signal.category
        category_unit_ids[category].add(unit.unit_id)
        category_signal_counts[category][signal.signal] += 1

        example_key = (unit.unit_id, signal.signal)
        if example_key in category_example_keys[category]:
            return
        if len(category_examples[category]) >= MAX_EXAMPLES_PER_CATEGORY:
            return

        category_example_keys[category].add(example_key)
        category_examples[category].append(
            {
                "source_type": unit.source_type,
                "video_id": unit.video_id,
                "source_file": unit.source_file,
                "field_name": unit.field_name,
                "line_number": unit.line_number,
                "signal": signal.signal,
                "reason": signal.reason,
                "excerpt": _format_excerpt(unit.text),
            }
        )

    for unit in text_units:
        for detector in DETECTORS:
            for signal in detector(unit):
                register_signal(unit, signal)

    variant_families = _collect_variant_families(text_units)
    for family in variant_families:
        family_reason = f"Variant forms detected: {', '.join(family.forms[:3])}"
        for unit_id in family.unit_ids:
            unit = unit_index[unit_id]
            register_signal(
                unit,
                DetectionSignal(
                    category="orthographic_variation",
                    signal="variant_token_family",
                    reason=family_reason,
                ),
            )

    total_text_units = len(text_units)
    sorted_categories = sorted(
        category_unit_ids.keys(),
        key=lambda category: (-len(category_unit_ids[category]), category),
    )

    frequency_table: list[dict[str, object]] = []
    category_details: dict[str, dict[str, object]] = {}

    for category in sorted_categories:
        count = len(category_unit_ids[category])
        percentage = (
            round((count / total_text_units) * 100, 2) if total_text_units else 0.0
        )
        signal_breakdown = dict(
            sorted(
                category_signal_counts[category].items(),
                key=lambda item: (-item[1], item[0]),
            )
        )

        frequency_table.append(
            {
                "category": category,
                "count": count,
                "percentage": percentage,
            }
        )
        category_details[category] = {
            "count": count,
            "percentage": percentage,
            "signal_breakdown": signal_breakdown,
            "examples": category_examples[category],
        }

    top_variant_families = [
        {
            "normalized": family.normalized,
            "forms": list(family.forms),
            "unit_count": len(family.unit_ids),
        }
        for family in variant_families[:20]
    ]

    report: dict[str, object] = {
        "generated_at": datetime.now().isoformat(),
        "analysis_scope": {
            "code_switching_rule": "any_latin_token_present",
            "required_categories": [
                "punctuation_inconsistency",
                "code_switching",
                "orthographic_variation",
                "formatting_artifacts",
            ],
            "extra_categories_allowed": True,
        },
        "input_summary": input_summary,
        "detected_categories": sorted_categories,
        "frequency_table": frequency_table,
        "category_details": category_details,
        "orthographic_variant_families": top_variant_families,
    }
    return report


def _write_json_report(output_path: Path, report: dict[str, object]) -> None:
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, ensure_ascii=False, indent=2)


def _write_markdown_summary(output_path: Path, report: dict[str, object]) -> None:
    generated_at = str(report.get("generated_at", ""))
    input_summary = report.get("input_summary", {})
    frequency_table = report.get("frequency_table", [])
    category_details = report.get("category_details", {})
    detected_categories = report.get("detected_categories", [])
    variant_families = report.get("orthographic_variant_families", [])

    lines: list[str] = [
        "# Pattern and Noise Detection Summary",
        "",
        f"Generated: {generated_at}",
        "",
        "## Dataset Coverage",
        "",
        f"- Transcript files: {input_summary.get('transcript_files', 0)}",
        f"- QA records: {input_summary.get('qa_records', 0)}",
        f"- Text units analyzed: {input_summary.get('text_units_total', 0)}",
        "",
        "## Irregularity Frequency Table",
        "",
        "| Category | Count | Percentage of Text Units |",
        "|---|---:|---:|",
    ]

    for row in frequency_table:
        category = row.get("category", "")
        count = row.get("count", 0)
        percentage = float(row.get("percentage", 0.0))
        lines.append(f"| {category} | {count} | {percentage:.2f}% |")

    lines.extend(
        [
            "",
            "## Example Irregularity Showcase",
            "",
        ]
    )

    for category in detected_categories:
        details = category_details.get(category, {})
        count = details.get("count", 0)
        percentage = float(details.get("percentage", 0.0))
        signal_breakdown = details.get("signal_breakdown", {})
        examples = details.get("examples", [])

        lines.extend(
            [
                f"### {category}",
                "",
                f"- Count: {count} ({percentage:.2f}% of text units)",
                "",
                "| Signal | Count |",
                "|---|---:|",
            ]
        )

        for signal_name, signal_count in signal_breakdown.items():
            lines.append(f"| {signal_name} | {signal_count} |")

        lines.append("")
        if not examples:
            lines.append("No examples captured for this category.")
            lines.append("")
            continue

        for index, example in enumerate(examples, start=1):
            source_type = example.get("source_type", "")
            video_id = example.get("video_id", "")
            source_file = example.get("source_file", "")
            line_number = example.get("line_number")
            signal = example.get("signal", "")
            reason = example.get("reason", "")
            excerpt = example.get("excerpt", "")

            location = f"{source_type} | {video_id} | {source_file}"
            if line_number is not None:
                location += f" | line {line_number}"

            lines.append(f"{index}. **Location:** {location}")
            lines.append(f"   - **Signal:** {signal}")
            lines.append(f"   - **Reason:** {reason}")
            lines.append(f"   - **Excerpt:** {excerpt}")

        lines.append("")

    if variant_families:
        lines.extend(
            [
                "## Orthographic Variant Family Highlights",
                "",
                "| Normalized Form | Surface Forms | Unit Count |",
                "|---|---|---:|",
            ]
        )
        for family in variant_families[:10]:
            normalized = family.get("normalized", "")
            forms = ", ".join(family.get("forms", []))
            unit_count = family.get("unit_count", 0)
            lines.append(f"| {normalized} | {forms} | {unit_count} |")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_irregularity_detection(paths: Optional[MS1Paths] = None) -> dict[str, object]:
    if paths is None:
        paths = resolve_ms1_paths()

    text_units, input_summary = _build_text_units(paths)
    if not text_units:
        raise ValueError("No text units were loaded for irregularity detection.")

    report = _analyze_irregularities(text_units, input_summary)

    output_dir = paths.experiments_ms1
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / JSON_ARTIFACT_NAME
    markdown_path = output_dir / MARKDOWN_ARTIFACT_NAME

    _write_json_report(json_path, report)
    _write_markdown_summary(markdown_path, report)

    print(f"Saved irregularity report: {json_path}")
    print(f"Saved markdown summary: {markdown_path}")
    return report


if __name__ == "__main__":
    run_irregularity_detection()
