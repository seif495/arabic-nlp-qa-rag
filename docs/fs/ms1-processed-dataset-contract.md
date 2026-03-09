# MS1 Processed Dataset Export Contract

Canonical implementation:

- `src/ms1/dataset_export.py`

Export destination:

- `data/processed/ms1/ms1_dataset_processed_v001.jsonl`

Sanity check destination:

- `data/processed/ms1/ms1_dataset_validation_summary_v001.json`

## Record schema

Each JSONL line must follow `ProcessedDatasetRecord` from `src/common/schemas.py`:

| Field | Type | Description |
| --- | --- | --- |
| `sample_id` | `str` | Stable sample identifier |
| `transcript_id` | `str` | Source transcript ID |
| `qa_id` | `str` | Source QA identifier |
| `cleaned_record_id` | `str` | Source cleaned-record identifier |
| `normalized_context` | `str` | Cleaned and normalized transcript text |
| `question_text` | `str` | Question used for supervision |
| `answer_text` | `str` | Ground-truth answer text |
| `split` | `str` | Dataset split (`train`, `val`, `test`) |

## Linkage requirements

- `transcript_id`, `qa_id`, and `cleaned_record_id` are required and non-empty.
- QA-to-transcript linkage is preserved by validating transcript consistency before export.

## Sanity-check requirements

The validation summary must include:

- `record_count`
- `split_count`
- `linkage_ok`
- `usable`

`usable` must be `true` only when exported records are non-empty and linkage checks pass.
