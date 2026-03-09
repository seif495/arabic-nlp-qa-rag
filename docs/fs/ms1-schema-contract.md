# MS1 Schema Contract

Canonical implementation:

- `src/common/schemas.py`

Schema example artifacts:

- `docs/fs/artifacts/ms1/ms1-transcript-record.example.json`
- `docs/fs/artifacts/ms1/ms1-qa-record.example.json`
- `docs/fs/artifacts/ms1/ms1-cleaned-record.example.json`
- `docs/fs/artifacts/ms1/ms1-prepared-dataset-sample.example.json`
- `docs/fs/artifacts/ms1/ms1-processed-dataset.sample.jsonl`

## Field dictionary

| Schema | Field | Type | Description |
| --- | --- | --- | --- |
| TranscriptRecord | transcript_id | str | Stable ID for transcript unit |
| TranscriptRecord | source_path | str | Source file path relative to repo |
| TranscriptRecord | speaker_label | str | Speaker name or role |
| TranscriptRecord | raw_text | str | Raw transcript text before cleaning |
| TranscriptRecord | language | str | Language code (for MS1 use `ar`) |
| QARecord | qa_id | str | Stable ID for QA entry |
| QARecord | transcript_id | str | Transcript ID associated with QA |
| QARecord | question_text | str | Question text |
| QARecord | answer_text | str | Ground-truth answer text |
| QARecord | answer_start_char | int | Character offset for answer in context |
| CleanedRecord | record_id | str | Stable ID for cleaned record |
| CleanedRecord | transcript_id | str | Source transcript ID |
| CleanedRecord | cleaned_text | str | Text after cleaning operations |
| CleanedRecord | normalized_text | str | Text after normalization operations |
| PreparedDatasetSample | sample_id | str | Stable ID for model-ready sample |
| PreparedDatasetSample | qa_id | str | Source QA ID |
| PreparedDatasetSample | transcript_id | str | Source transcript ID |
| PreparedDatasetSample | input_text | str | Prompt or context sent to model |
| PreparedDatasetSample | target_text | str | Expected output string |
| PreparedDatasetSample | split | str | Dataset split label (train/val/test) |
| ProcessedDatasetRecord | sample_id | str | Stable ID for exported sample |
| ProcessedDatasetRecord | transcript_id | str | Source transcript ID |
| ProcessedDatasetRecord | qa_id | str | Source QA ID |
| ProcessedDatasetRecord | cleaned_record_id | str | Source cleaned record ID |
| ProcessedDatasetRecord | normalized_context | str | Normalized context for training |
| ProcessedDatasetRecord | question_text | str | Supervision question text |
| ProcessedDatasetRecord | answer_text | str | Supervision answer text |
| ProcessedDatasetRecord | split | str | Dataset split label (train/val/test) |
