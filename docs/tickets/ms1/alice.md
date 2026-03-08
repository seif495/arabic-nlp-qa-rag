# Milestone 1 Tickets - Alice

Total story points: 8

## MS1-INGEST-01: Raw Dataset Loader and Validation

### Description

Implement loading and validation for transcript and QA datasets.

The loader must parse all files, validate pairing, and return structured records using the shared schemas.

### Acceptance Criteria

- All 13 transcript files load successfully.
- All QA CSV files load successfully.
- Transcript to QA pairing validation exists.
- Invalid records surface readable errors.
- Loader outputs match shared schemas.

### Produces Artifacts

- dataset inventory table
- validation summary
- example malformed case (if any)
- `experiments/ms1/raw_loader_summary.json`

### Blocking Class

Hard blocker

### Notes

- Dependencies: `MS1-INFRA-02`
- Story points: 2

---

## MS1-ANALYSIS-01: Corpus Distribution Analysis

### Description

Compute descriptive statistics over transcripts and QA data.

This module analyzes structural properties of the dataset.

### Acceptance Criteria

- transcript length distribution computed
- question length distribution computed
- answer length distribution computed
- token and character summary statistics generated
- outputs saved in experiment artifacts

### Produces Artifacts

- transcript length histogram
- QA length histogram
- summary statistics table
- `experiments/ms1/corpus_stats.json`
- short observation notes

### Blocking Class

Non-blocker

### Notes

- Dependencies: `MS1-INGEST-01`
- Story points: 3

---

## MS1-ANALYSIS-02: Pattern and Noise Detection

### Description

Detect linguistic irregularities and structural noise in the dataset.

Examples include punctuation inconsistency, code-switching signals, orthographic variation, and formatting artifacts.

### Acceptance Criteria

- irregularity categories detected and summarized
- frequency table generated
- example instances stored
- results saved for report usage

### Produces Artifacts

- irregularity frequency table
- example irregularity showcase
- `experiments/ms1/irregularity_report.json`
- markdown summary of findings

### Blocking Class

Non-blocker

### Notes

- Dependencies: `MS1-INGEST-01`
- Story points: 3
