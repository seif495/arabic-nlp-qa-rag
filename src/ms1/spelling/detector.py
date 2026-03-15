"""
MS1-ANALYSIS-03: Spelling Inconsistency Detection

Identifies inconsistent spelling forms across the corpus and optionally
unifies high-confidence cases.  Ambiguous cases are documented but not changed.

Strategy
--------
Two words are treated as *variant spellings* of each other when they differ
**only** by one of the following character-level transformations:

1. Alef variants: أ / إ / آ / ٱ ↔ ا
2. Alef maqsura ↔ ya: ى ↔ ي
3. Teh marbuta ↔ ha: ة ↔ ه  (documented; conservative unification)
4. Presence/absence of tatweel: ـ
5. Presence/absence of diacritics

A *canonical form* is the normalized version (rules 1-4 applied from PROC-02
plus tatweel/diacritic removal).  Words that share a canonical form but differ
in the original are *inconsistent spelling variants*.

A variant is unified (changed) when ≥ 90 % of occurrences already use one
spelling.  All other cases are left as-is and noted in the ambiguity catalog.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.common.paths import MS1Paths, make_ms1_output_filename, resolve_ms1_paths
from src.ms1.cleaning.cleaner import clean_qa_text, clean_transcript
from src.ms1.ingest.loader import load_qa_files, load_transcripts
from src.ms1.normalization.normalizer import normalize_arabic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNIFICATION_THRESHOLD = 0.90  # dominant form must cover ≥ 90 % of occurrences

_TOKEN_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]+")  # Arabic-only tokens


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------


def _tokenize_arabic(text: str) -> list[str]:
    """Extract Arabic-only tokens from text."""
    return _TOKEN_RE.findall(text)


def _canonical(token: str) -> str:
    """Reduce a token to its canonical form for grouping."""
    return normalize_arabic(token)


def detect_spelling_inconsistencies(
    tokens: list[str],
    threshold: float = UNIFICATION_THRESHOLD,
) -> dict:
    """Detect inconsistent spelling forms in a token list.

    Args:
        tokens: Flat list of Arabic tokens (from a single document or corpus).
        threshold: Fraction of occurrences required to trigger unification.

    Returns:
        A dict with keys:
        - ``groups``: list of variant groups with counts and decision.
        - ``unified``: {original_form: canonical_form} for high-confidence cases.
        - ``ambiguous``: list of groups left unchanged with rationale.
        - ``stats``: summary counts.
    """
    # Count raw token frequencies
    raw_freq: Counter[str] = Counter(tokens)

    # Group tokens by canonical form
    canonical_groups: dict[str, Counter[str]] = defaultdict(Counter)
    for token, count in raw_freq.items():
        canonical_groups[_canonical(token)][token] += count

    # Only care about canonical groups that have ≥ 2 distinct surface forms
    inconsistent = {
        canon: variants
        for canon, variants in canonical_groups.items()
        if len(variants) > 1
    }

    unified: dict[str, str] = {}
    ambiguous_groups: list[dict] = []
    all_groups: list[dict] = []

    for canon, variants in sorted(inconsistent.items()):
        total = sum(variants.values())
        dominant_form, dominant_count = variants.most_common(1)[0]
        dominant_frac = dominant_count / total

        variant_list = [
            {"form": form, "count": count, "pct": round(100.0 * count / total, 1)}
            for form, count in variants.most_common()
        ]

        group_entry = {
            "canonical": canon,
            "total_occurrences": total,
            "dominant_form": dominant_form,
            "dominant_pct": round(100.0 * dominant_frac, 1),
            "variants": variant_list,
        }

        if dominant_frac >= threshold:
            # High-confidence: unify minority forms → dominant form
            for form in variants:
                if form != dominant_form:
                    unified[form] = dominant_form
            group_entry["decision"] = "unified"
        else:
            ambiguous_groups.append(
                {
                    **group_entry,
                    "reason": (
                        f"No single form exceeds {threshold * 100:.0f}% "
                        f"(dominant: {dominant_frac * 100:.1f}%)"
                    ),
                }
            )
            group_entry["decision"] = "ambiguous"

        all_groups.append(group_entry)

    return {
        "groups": all_groups,
        "unified": unified,
        "ambiguous": ambiguous_groups,
        "stats": {
            "canonical_groups_with_variants": len(inconsistent),
            "unified_groups": len(all_groups) - len(ambiguous_groups),
            "ambiguous_groups": len(ambiguous_groups),
            "total_minority_forms_unified": len(unified),
        },
    }


def apply_unification(text: str, unified_map: dict[str, str]) -> str:
    """Replace minority spelling forms in *text* using the unified map.

    Only whole-token matches are replaced to avoid partial-word substitution.
    """
    if not unified_map:
        return text

    def _replace(match: re.Match) -> str:  # type: ignore[type-arg]
        token = match.group()
        return unified_map.get(token, token)

    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in unified_map) + r")\b")
    return pattern.sub(_replace, text)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_spelling_analysis(
    paths: MS1Paths | None = None,
    repo_root: Path | None = None,
) -> dict:
    """Run spelling inconsistency detection across the full corpus.

    Saves four artifacts to ``experiments/ms1/``:

    * ``ms1_spelling_catalog_v001.json``            – full inconsistency catalog
    * ``ms1_spelling_unification_examples_v001.json`` – unified form examples
    * ``ms1_spelling_changed_unchanged_v001.json``  – changed vs unchanged examples
    * ``ms1_spelling_ambiguity_notes_v001.md``      – ambiguity documentation

    Returns the summary dict.
    """
    resolved_paths = paths or resolve_ms1_paths(repo_root=repo_root)
    transcripts = load_transcripts(resolved_paths)
    qa_records = load_qa_files(resolved_paths)

    # Collect all Arabic tokens from cleaned transcripts + QA text
    all_tokens: list[str] = []
    for transcript in transcripts:
        cleaned = clean_transcript(transcript["content"])
        all_tokens.extend(_tokenize_arabic(cleaned))
    for record in qa_records:
        for field in ("question", "answer"):
            cleaned = clean_qa_text(record.get(field, ""))
            all_tokens.extend(_tokenize_arabic(cleaned))

    result = detect_spelling_inconsistencies(all_tokens)
    unified_map = result["unified"]
    ambiguous_groups = result["ambiguous"]
    all_groups = result["groups"]
    stats = result["stats"]

    # Build changed vs unchanged examples (sample transcripts before/after unification)
    changed_examples: list[dict] = []
    unchanged_examples: list[dict] = []

    for transcript in transcripts[:5]:
        cleaned = clean_transcript(transcript["content"])
        unified_text = apply_unification(cleaned, unified_map)
        if cleaned != unified_text:
            changed_examples.append(
                {
                    "transcript_id": transcript["video_id"],
                    "before_sample": cleaned[:300],
                    "after_sample": unified_text[:300],
                }
            )
        else:
            unchanged_examples.append(
                {
                    "transcript_id": transcript["video_id"],
                    "note": "no high-confidence unifications applied",
                    "sample": cleaned[:200],
                }
            )

    output_dir = resolved_paths.experiments_ms1

    # Artifact 1: full spelling inconsistency catalog
    catalog_path = output_dir / make_ms1_output_filename(
        "spelling", "catalog", 1, "json"
    )
    with catalog_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"stats": stats, "groups": all_groups},
            fh,
            indent=2,
            ensure_ascii=False,
        )

    # Artifact 2: unification example set
    unification_path = output_dir / make_ms1_output_filename(
        "spelling", "unification_examples", 1, "json"
    )
    unified_examples = [
        {"minority_form": form, "dominant_form": dominant}
        for form, dominant in list(unified_map.items())[:50]
    ]
    with unification_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"unified_map_size": len(unified_map), "examples": unified_examples},
            fh,
            indent=2,
            ensure_ascii=False,
        )

    # Artifact 3: changed vs unchanged examples
    changed_path = output_dir / make_ms1_output_filename(
        "spelling", "changed_unchanged", 1, "json"
    )
    with changed_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "changed_examples": changed_examples,
                "unchanged_examples": unchanged_examples,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    # Artifact 4: ambiguity notes
    ambiguity_path = output_dir / make_ms1_output_filename(
        "spelling", "ambiguity_notes", 1, "md"
    )
    _write_ambiguity_notes(ambiguity_path, ambiguous_groups, stats)

    summary = {
        **stats,
        "corpus_tokens_analyzed": len(all_tokens),
        "unification_threshold": UNIFICATION_THRESHOLD,
    }
    return summary


# ---------------------------------------------------------------------------
# Artifact helper
# ---------------------------------------------------------------------------


def _write_ambiguity_notes(
    path: Path,
    ambiguous_groups: list[dict],
    stats: dict,
) -> None:
    lines: list[str] = [
        "# MS1 Spelling Ambiguity Notes\n",
        "## Overview\n",
        f"Total canonical groups with variants: {stats['canonical_groups_with_variants']}  ",
        f"Unified (high-confidence): {stats['unified_groups']}  ",
        f"Left unchanged (ambiguous): {stats['ambiguous_groups']}\n",
        "\nA group is **ambiguous** when no single spelling form covers ≥ 90 % of "
        "occurrences.  These cases are documented below but **not changed** to "
        "preserve author intent and avoid introducing errors.\n",
        "\n---\n",
        "## Ambiguous Groups\n",
    ]

    if not ambiguous_groups:
        lines.append("_No ambiguous groups found._\n")
    else:
        for i, group in enumerate(ambiguous_groups[:30], 1):
            lines.append(f"### {i}. Canonical form: `{group['canonical']}`\n")
            lines.append(f"- Total occurrences: {group['total_occurrences']}")
            lines.append(
                f"- Dominant form: `{group['dominant_form']}` "
                f"({group['dominant_pct']} %)"
            )
            lines.append("- Variants:")
            for v in group["variants"]:
                lines.append(f"  - `{v['form']}`: {v['count']} ({v['pct']} %)")
            lines.append(f"- Reason not unified: {group['reason']}\n")

    path.write_text("\n".join(lines), encoding="utf-8")
