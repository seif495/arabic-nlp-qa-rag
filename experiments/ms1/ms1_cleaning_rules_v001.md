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
| Pattern | `^\d+(?:\.\d+)?:\s*` |
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
