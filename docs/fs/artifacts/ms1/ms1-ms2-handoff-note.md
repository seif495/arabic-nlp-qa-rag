# MS1 to MS2 Handoff Note

- Ticket: `MS1-INFRA-04`
- Export file: `data/processed/ms1/ms1_dataset_processed_v001.jsonl`
- Validation summary: `data/processed/ms1/ms1_dataset_validation_summary_v001.json`

## Handoff Guarantees

- Each exported record preserves transcript, QA, and cleaned-record linkage via IDs.
- Exported records include normalized context, question text, answer text, and split.
- A sanity-check summary is generated for downstream ingestion validation.
