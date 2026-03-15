# MS1 Arabic Normalization Policy

## Overview

This document defines the Arabic character normalization policy applied in
Milestone 1. Normalization is performed **after** the deterministic cleaning
step (MS1-PROC-01) and operates at the character level.

## Rule Table

| #   | Rule               | Input → Output                | Unicode                  |
| --- | ------------------ | ----------------------------- | ------------------------ |
| 1   | Tatweel removal    | ـ → (deleted)                 | U+0640                   |
| 2   | Diacritic removal  | ً ٌ ٍ َ ُ ِ ّ ْ ٰ → (deleted) | U+064B–U+0652, U+0670    |
| 3   | Alef normalisation | أ إ آ ٱ → ا                   | U+0623/25/22/71 → U+0627 |
| 4   | Alef maqsura → ya  | ى → ي                         | U+0649 → U+064A          |

## Design Rationale

**Rule 1 – Tatweel removal**
Tatweel (kashida) is a cosmetic elongation mark that carries no lexical
meaning and creates vocabulary fragmentation (`كتاب` ≠ `كتـاب`).

**Rule 2 – Diacritic removal**
Harakat (short vowel marks) are absent in most Arabic text corpora and
inconsistently present in transcripts. Removing them aligns the transcript
vocabulary with the likely encoding of downstream models.

**Rule 3 – Alef normalisation**
`أ`, `إ`, `آ`, and `ٱ` are all pronounced / written as the same base
letter in unvocalised Arabic. Normalising them reduces sparsity without
semantic loss for Egyptian-dialect text.

**Rule 4 – Alef maqsura → ya**
`ى` and `ي` are frequently confused in informal/dialectal writing.
Unifying to `ي` reduces vocabulary size and prevents artificial splits.

## Scope

Rules are applied to:

- All cleaned transcript texts before writing to `data/interim/normalized/`.
- All QA question and answer fields before dataset export.

## Reproducibility

The `normalize_arabic()` function accepts keyword flags for each rule,
making the exact policy fully reproducible and inspectable.
