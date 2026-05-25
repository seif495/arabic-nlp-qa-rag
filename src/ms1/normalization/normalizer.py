"""
MS1-PROC-02: Arabic Character Normalization

Implements character-level normalization rules for Arabic text and documents
the chosen policy and associated risks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.common.paths import MS1Paths, make_ms1_output_filename, resolve_ms1_paths
from src.ms1.cleaning.cleaner import clean_qa_text, clean_transcript
from src.ms1.ingest.loader import load_qa_files, load_transcripts

# ---------------------------------------------------------------------------
# Normalization tables
# ---------------------------------------------------------------------------

# Alef variants → plain alef (U+0627 ا)
# أ U+0623, إ U+0625, آ U+0622, ٱ U+0671
_ALEF_MAP: dict[str, str] = {
    "\u0623": "\u0627",
    "\u0625": "\u0627",
    "\u0622": "\u0627",
    "\u0671": "\u0627",
}

# Alef maqsura → ya  ى U+0649 → ي U+064A
_YA_MAP: dict[str, str] = {
    "\u0649": "\u064a",
}

# Tatweel / kashida U+0640
_TATWEEL = "\u0640"

# Arabic diacritics (harakat / tashkeel): U+064B–U+0652 + superscript alef U+0670
_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670]")

# Pre-compiled character-class patterns for bulk replacement
_ALEF_RE = re.compile("|".join(re.escape(k) for k in _ALEF_MAP))
_YA_RE = re.compile("|".join(re.escape(k) for k in _YA_MAP))


# ---------------------------------------------------------------------------
# Core normalization function
# ---------------------------------------------------------------------------


def normalize_arabic(
    text: str,
    *,
    remove_diacritics: bool = True,
    normalize_alef: bool = True,
    normalize_ya: bool = True,
    remove_tatweel: bool = True,
) -> str:
    """Normalize Arabic text using configurable character-level rules.

    Args:
        text: Input Arabic string (may contain mixed Arabic/Latin).
        remove_diacritics: Strip harakat (U+064B–U+0652, U+0670).
        normalize_alef: Map أ إ آ ٱ → ا.
        normalize_ya: Map alef maqsura ى → ي.
        remove_tatweel: Remove kashida ـ (U+0640).

    Returns:
        Normalized string.  Non-Arabic characters are left unchanged.
    """
    if remove_tatweel:
        text = text.replace(_TATWEEL, "")
    if remove_diacritics:
        text = _DIACRITICS_RE.sub("", text)
    if normalize_alef:
        text = _ALEF_RE.sub(lambda m: _ALEF_MAP[m.group()], text)
    if normalize_ya:
        text = _YA_RE.sub(lambda m: _YA_MAP[m.group()], text)
    return text


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\S+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _count_changed_tokens(raw_tokens: list[str], norm_tokens: list[str]) -> int:
    return sum(1 for r, n in zip(raw_tokens, norm_tokens) if r != n)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_normalization_pipeline(
    paths: MS1Paths | None = None,
    repo_root: Path | None = None,
) -> dict:
    """Run Arabic normalization on all cleaned transcripts and QA records.

    Loads transcripts and QA data, cleans them first (PROC-01), then applies
    normalization (PROC-02).  Saves four artifacts to ``experiments/ms1/``:

    * ``ms1_normalization_policy_v001.md``          – policy + rule table
    * ``ms1_normalization_token_comparison_v001.json`` – per-transcript before/after counts
    * ``ms1_normalization_examples_v001.json``      – example normalized outputs
    * ``ms1_normalization_risk_v001.md``            – risk analysis note

    Normalized text is also written per-transcript to
    ``data/interim/normalized/`` as UTF-8 ``.txt`` files.

    Returns the summary dict.
    """
    resolved_paths = paths or resolve_ms1_paths(repo_root=repo_root)
    transcripts = load_transcripts(resolved_paths)
    qa_records = load_qa_files(resolved_paths)

    interim_dir = resolved_paths.data_interim / "normalized"
    interim_dir.mkdir(parents=True, exist_ok=True)

    token_comparison: list[dict] = []
    examples: list[dict] = []
    total_tokens = 0
    total_changed = 0

    for transcript in transcripts:
        cleaned = clean_transcript(transcript["content"])
        normalized = normalize_arabic(cleaned)

        # Write interim file
        out_file = interim_dir / f"{transcript['video_id']}_normalized.txt"
        out_file.write_text(normalized, encoding="utf-8")

        raw_tokens = _tokenize(cleaned)
        norm_tokens = _tokenize(normalized)
        n_tokens = len(raw_tokens)
        n_changed = _count_changed_tokens(raw_tokens, norm_tokens)
        total_tokens += n_tokens
        total_changed += n_changed

        token_comparison.append(
            {
                "transcript_id": transcript["video_id"],
                "token_count": n_tokens,
                "tokens_changed": n_changed,
                "change_pct": round(100.0 * n_changed / max(n_tokens, 1), 2),
            }
        )

        if len(examples) < 5:
            # Find the first token that actually changed
            changed_pairs = [
                {"before": r, "after": n}
                for r, n in zip(raw_tokens, norm_tokens)
                if r != n
            ][:10]
            examples.append(
                {
                    "transcript_id": transcript["video_id"],
                    "cleaned_sample": cleaned[:300],
                    "normalized_sample": normalized[:300],
                    "changed_token_examples": changed_pairs,
                }
            )

    # QA normalization stats
    qa_changed_tokens = 0
    qa_total_tokens = 0
    for record in qa_records:
        for field in ("question", "answer"):
            cleaned = clean_qa_text(record.get(field, ""))
            normalized = normalize_arabic(cleaned)
            raw_toks = _tokenize(cleaned)
            norm_toks = _tokenize(normalized)
            qa_total_tokens += len(raw_toks)
            qa_changed_tokens += _count_changed_tokens(raw_toks, norm_toks)

    summary: dict = {
        "transcripts_processed": len(transcripts),
        "qa_records_processed": len(qa_records),
        "transcript_total_tokens": total_tokens,
        "transcript_tokens_changed": total_changed,
        "transcript_change_pct": round(100.0 * total_changed / max(total_tokens, 1), 2),
        "qa_total_tokens": qa_total_tokens,
        "qa_tokens_changed": qa_changed_tokens,
        "qa_change_pct": round(100.0 * qa_changed_tokens / max(qa_total_tokens, 1), 2),
        "rules_applied": [
            "remove_tatweel",
            "remove_diacritics",
            "normalize_alef",
            "normalize_ya",
        ],
        "per_transcript": token_comparison,
    }

    output_dir = resolved_paths.experiments_ms1

    # Artifact 1: policy note
    policy_path = output_dir / make_ms1_output_filename(
        "normalization", "policy", 1, "md"
    )
    _write_policy_doc(policy_path)

    # Artifact 2: token comparison
    comparison_path = output_dir / make_ms1_output_filename(
        "normalization", "token_comparison", 1, "json"
    )
    with comparison_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # Artifact 3: example normalized outputs
    examples_path = output_dir / make_ms1_output_filename(
        "normalization", "examples", 1, "json"
    )
    with examples_path.open("w", encoding="utf-8") as fh:
        json.dump(examples, fh, indent=2, ensure_ascii=False)

    # Artifact 4: risk analysis
    risk_path = output_dir / make_ms1_output_filename("normalization", "risk", 1, "md")
    _write_risk_doc(risk_path)

    return summary


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


def _write_policy_doc(path: Path) -> None:
    content = """\
# MS1 Arabic Normalization Policy

## Overview

This document defines the Arabic character normalization policy applied in
Milestone 1.  Normalization is performed **after** the deterministic cleaning
step (MS1-PROC-01) and operates at the character level.

## Rule Table

| # | Rule | Input → Output | Unicode |
|---|------|----------------|---------|
| 1 | Tatweel removal | ـ → (deleted) | U+0640 |
| 2 | Diacritic removal | ً ٌ ٍ َ ُ ِ ّ ْ ٰ → (deleted) | U+064B–U+0652, U+0670 |
| 3 | Alef normalisation | أ إ آ ٱ → ا | U+0623/25/22/71 → U+0627 |
| 4 | Alef maqsura → ya | ى → ي | U+0649 → U+064A |

## Design Rationale

**Rule 1 – Tatweel removal**
Tatweel (kashida) is a cosmetic elongation mark that carries no lexical
meaning and creates vocabulary fragmentation (``كتاب`` ≠ ``كتـاب``).

**Rule 2 – Diacritic removal**
Harakat (short vowel marks) are absent in most Arabic text corpora and
inconsistently present in transcripts.  Removing them aligns the transcript
vocabulary with the likely encoding of downstream models.

**Rule 3 – Alef normalisation**
``أ``, ``إ``, ``آ``, and ``ٱ`` are all pronounced / written as the same base
letter in unvocalised Arabic.  Normalising them reduces sparsity without
semantic loss for Egyptian-dialect text.

**Rule 4 – Alef maqsura → ya**
``ى`` and ``ي`` are frequently confused in informal/dialectal writing.
Unifying to ``ي`` reduces vocabulary size and prevents artificial splits.

## Scope

Rules are applied to:
- All cleaned transcript texts before writing to ``data/interim/normalized/``.
- All QA question and answer fields before dataset export.

## Reproducibility

The ``normalize_arabic()`` function accepts keyword flags for each rule,
making the exact policy fully reproducible and inspectable.
"""
    path.write_text(content, encoding="utf-8")


def _write_risk_doc(path: Path) -> None:
    content = """\
# MS1 Normalization Risk Analysis

## High-Confidence (Low-Risk) Rules

| Rule | Risk Level | Justification |
|------|-----------|---------------|
| Tatweel removal | Low | Pure cosmetic; no lexical or semantic role |
| Diacritic removal | Low | Consistent with unvocalised Arabic; models trained on raw web text do not expect harakat |

## Moderate-Risk Rules

| Rule | Risk | Mitigation |
|------|------|------------|
| Alef normalisation | Moderate | Rare loan words or proper nouns may rely on hamza placement for disambiguation. Effect is negligible in the predominantly colloquial ElDa7ee7 corpus. |
| Alef maqsura → ya | Moderate | In classical Arabic, ى at word-end is semantically distinct from ي. In Egyptian colloquial Arabic (this corpus), the distinction is rarely maintained in writing. |

## Extractive QA Risk

The dataset uses **extractive** QA: the answer is a direct span from the
transcript.  If normalization is applied to **both** the transcript context
and the QA answers, alignment is preserved.  The export pipeline (INFRA-04)
must apply identical normalization to both fields.

## Recommendation

Apply all four rules uniformly to transcripts and QA text.  Keep the raw
``data/external/`` files unmodified so that the original alignment can be
recovered if needed.
"""
    path.write_text(content, encoding="utf-8")
