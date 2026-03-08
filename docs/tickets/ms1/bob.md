# Milestone 1 Tickets - Bob

Total story points: 9

## MS1-PROC-01: Deterministic Cleaning Pipeline

### Description

Implement conservative deterministic cleaning rules for transcripts and QA text.

Cleaning should remove obvious noise while preserving semantic integrity.

### Acceptance Criteria

- whitespace normalization implemented
- formatting artifacts handled
- cleaning deterministic and repeatable
- before/after comparisons saved

### Produces Artifacts

- cleaning rules document
- before/after examples
- cleaning effect summary table

### Blocking Class

Non-blocker

### Notes

- Dependencies: `MS1-INGEST-01`
- Story points: 3

---

## MS1-PROC-02: Arabic Character Normalization

### Description

Implement normalization rules for Arabic text and document the chosen policy.

### Acceptance Criteria

- normalization rules documented
- normalization implemented as reusable function
- pipeline applies normalization across all records
- before/after counts generated

### Produces Artifacts

- normalization policy note
- before/after token comparison
- example normalized outputs
- normalization risk analysis note

### Blocking Class

Non-blocker

### Notes

- Dependencies: `MS1-INGEST-01`
- Story points: 3

---

## MS1-ANALYSIS-03: Spelling Inconsistency Detection

### Description

Identify inconsistent spelling forms and optionally unify high-confidence cases.

Focus on analysis first and conservative correction.

### Acceptance Criteria

- inconsistent forms detected
- high-confidence unifications implemented
- ambiguous cases documented but not changed
- results saved with examples

### Produces Artifacts

- spelling inconsistency catalog
- unification example set
- changed vs unchanged examples
- ambiguity notes

### Blocking Class

Non-blocker

### Notes

- Dependencies: `MS1-PROC-02`
- Story points: 3
