# Milestone 1 Tickets - Charly

Total story points: 11

## MS1-INFRA-01: Repository Paths and Artifact Layout [DONE]

### Description

Define canonical repository paths and artifact layout for Milestone 1 outputs.
All modules must rely on centralized path helpers rather than ad-hoc relative paths.

### Acceptance Criteria

- Shared path utilities resolve:
  - `data/external`
  - `data/interim`
  - `data/processed/ms1`
  - `experiments/ms1`
- All milestone modules import path helpers.
- Artifact filenames follow a documented naming convention.
- No hardcoded relative paths remain in milestone modules.

### Produces Artifacts

- `docs/fs/ms1-artifact-map.md`
- example output directory snapshot
- path resolution example output

### Blocking Class

Hard blocker

### Notes

- Dependencies: none
- Story points: 1

---

## MS1-INFRA-02: Core Data Structures and Python Schemas [DONE]

### Description

Define the internal data structures used across Milestone 1 and implement them as importable Python schemas.

Schemas must support:

- transcript records
- QA records
- cleaned and normalized text
- prepared dataset samples

Schemas must exist as Python code that can be imported by all modules.

### Acceptance Criteria

- `src/common/schemas.py` defines all shared data structures.
- Transcript schema implemented.
- QA schema implemented.
- Cleaned record schema implemented.
- Prepared dataset sample schema implemented.
- Example objects exist for each schema.

### Produces Artifacts

- `src/common/schemas.py`
- `docs/fs/ms1-schema-contract.md`
- JSON examples for each schema
- small field dictionary table for report usage

### Blocking Class

Hard blocker

### Notes

- Dependencies: `MS1-INFRA-01`
- Story points: 2

---

## MS1-INFRA-03: Milestone 1 CLI Interface

### Description

Define and implement the CLI entrypoints used to execute all Milestone 1 operations.

### Acceptance Criteria

The following commands exist and run:

```bash
python -m src.cli.ms1 profile
python -m src.cli.ms1 detect-irregularities
python -m src.cli.ms1 normalize
python -m src.cli.ms1 build-dataset
python -m src.cli.ms1 run-all
```

CLI files orchestrate modules rather than implementing business logic.

### Produces Artifacts

- CLI help output snapshot
- example execution logs
- `docs/fs/ms1-cli-contract.md`

### Blocking Class

Soft blocker

### Notes

- Dependencies: `MS1-INFRA-01`, `MS1-INFRA-02` [DONE]
- Story points: 2

---

## MS1-INFRA-04: Prepared Dataset for Downstream Modeling

### Description

Export cleaned and normalized dataset into a structured format suitable for later neural modeling.

### Acceptance Criteria

- processed dataset schema implemented
- transcript and QA linkage preserved
- dataset exported to `data/processed/ms1`
- export format documented
- sanity check confirms dataset usability

### Produces Artifacts

- processed dataset sample file
- export schema document
- dataset validation summary
- MS2 handoff note

### Blocking Class

Soft blocker

### Notes

- Dependencies: `MS1-INFRA-02` [DONE], `MS1-PROC-01`, `MS1-PROC-02`
- Story points: 3

---

## MS1-INFRA-05: End-to-End Milestone Pipeline

### Description

Integrate all Milestone 1 modules into a reproducible end-to-end pipeline executed through CLI.

### Acceptance Criteria

- `run-all` executes full pipeline
- outputs written to canonical directories
- pipeline runs from clean repository checkout
- execution logs generated

### Produces Artifacts

- full pipeline execution log
- artifact manifest
- integration checklist
- known limitations note

### Blocking Class

Hard blocker

### Notes

- Dependencies: all previous MS1 feature tickets
- Story points: 1

---

## MS1-REPORT-01: Milestone 1 Technical Report and Artifact Curation

### Description

Write the Milestone 1 report using artifacts produced throughout the pipeline.

The report explains design choices, insights, analysis results, and system limitations.

### Acceptance Criteria

- report includes graphs and tables from profiling
- irregularity findings explained
- normalization and cleaning decisions justified
- dataset preparation explained
- limitations documented
- report length approximately two pages

### Produces Artifacts

- `docs/reports/ms1_report.md`
- curated artifact bundle
- final figure selection
- engineering insight notes

### Blocking Class

Hard blocker

### Notes

- Dependencies: all MS1 tickets
- Story points: 2
