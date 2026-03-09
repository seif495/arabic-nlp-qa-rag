# Milestone 1 Artifact Map and Path Contracts

This document defines the canonical Milestone 1 (MS1) path and output contracts for ticket `MS1-INFRA-01`.

## 1) Scope

This contract covers:

- centralized path resolution requirements for all MS1 modules,
- canonical save locations for MS1 inputs/outputs,
- one agreed output filename convention,
- explicit rule against hardcoded ad hoc relative paths,
- a sample path manifest JSON and an example output tree snapshot.

## 2) Canonical directories

Path resolution policy:

- **Repository root source:** current working directory (`cwd`)
- **Path format in contracts:** repository-relative paths
- **Directory creation behavior:** writable output directories are created automatically

Canonical directories:

| Key                  | Canonical path        | Contract                           |
| -------------------- | --------------------- | ---------------------------------- |
| `data_external`      | `data/external/`      | Input only (read-only source data) |
| `data_interim`       | `data/interim/`       | Writable intermediate outputs      |
| `data_processed_ms1` | `data/processed/ms1/` | Writable finalized MS1 outputs     |
| `experiments_ms1`    | `experiments/ms1/`    | Writable MS1 experiment artifacts  |
| `docs_reports`       | `docs/reports/`       | Writable report outputs            |

## 3) Shared path helper contract (required usage)

All MS1 modules must use shared path helpers from `src/common/paths.py`.
No MS1 module is allowed to hardcode ad hoc relative paths for canonical locations.

Expected helper shape (typed dataclass bundle):

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MS1Paths:
    repo_root: Path
    data_external: Path
    data_interim: Path
    data_processed_ms1: Path
    experiments_ms1: Path
    docs_reports: Path


def resolve_ms1_paths(repo_root: Path | None = None) -> MS1Paths:
    """
    Resolve paths relative to cwd when repo_root is not provided.
    Create writable output directories if missing.
    """
```

## 4) Canonical output filename convention

All generated MS1 outputs must follow:

`ms1_<stage>_<name>_v###.<ext>`

Rules:

1. Entire filename body is `snake_case`.
2. `<stage>` must be one of:
   `ingest`, `profiling`, `cleaning`, `normalization`, `spelling`, `tokenization`, `dataset_build`, `evaluation`, `report`.
3. `<name>` is a concise artifact identifier (for example, `text_stats`, `normalized_transcripts`, `qa_split_train`).
4. `<ver>` is a zero-padded counter: `v001`, `v002`, ...
5. No ad hoc names such as `final.csv`, `new_final.json`, `test2_output.txt`.

Examples:

- `ms1_profiling_text_stats_v001.json`
- `ms1_normalization_normalized_transcripts_v001.parquet`
- `ms1_dataset_build_qa_split_train_v001.parquet`
- `ms1_evaluation_cleaning_metrics_v001.json`

## 5) No hardcoded ad hoc path rule

Non-compliant patterns:

```python
Path("data/processed/ms1") / "final.csv"
"../data/interim/tokenized/out.json"
```

Compliant pattern:

```python
paths = resolve_ms1_paths()
output_file = paths.data_processed_ms1 / "ms1_dataset_build_qa_split_train_v001.parquet"
```

## 6) Example output tree snapshot

Snapshot file: `docs/fs/artifacts/ms1/ms1-output-tree.snapshot.txt`

```text
data/
  external/
    transcripts/
    qa/
  interim/
    normalized/
      ms1_normalization_normalized_transcripts_v001.parquet
    tokenized/
      ms1_tokenization_token_sequences_v001.parquet
    splits/
      ms1_dataset_build_qa_split_train_v001.parquet
      ms1_dataset_build_qa_split_dev_v001.parquet
      ms1_dataset_build_qa_split_test_v001.parquet
  processed/
    ms1/
      ms1_cleaning_cleaned_corpus_v001.parquet
      ms1_profiling_text_stats_v001.json
experiments/
  ms1/
    ms1_evaluation_cleaning_metrics_v001.json
docs/
  reports/
    ms1_report_v001.md
```

## 7) Sample path manifest JSON

Path: `docs/fs/artifacts/ms1/ms1-path-manifest.sample.json`

This sample uses relative canonical paths only, matching the current contract.
