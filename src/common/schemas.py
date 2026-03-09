from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptRecord:
    transcript_id: str
    source_path: str
    speaker_label: str
    raw_text: str
    language: str


@dataclass(frozen=True)
class QARecord:
    qa_id: str
    transcript_id: str
    question_text: str
    answer_text: str
    answer_start_char: int


@dataclass(frozen=True)
class CleanedRecord:
    record_id: str
    transcript_id: str
    cleaned_text: str
    normalized_text: str


@dataclass(frozen=True)
class PreparedDatasetSample:
    sample_id: str
    qa_id: str
    transcript_id: str
    input_text: str
    target_text: str
    split: str


EXAMPLE_TRANSCRIPT_RECORD = TranscriptRecord(
    transcript_id="tr_0001",
    source_path="data/external/transcripts/episode_001.txt",
    speaker_label="Host",
    raw_text="الحلقة مقدمة عن المشروع",
    language="ar",
)

EXAMPLE_QA_RECORD = QARecord(
    qa_id="qa_0001",
    transcript_id="tr_0001",
    question_text="ما موضوع الحلقة؟",
    answer_text="الحلقة مقدمة عن المشروع",
    answer_start_char=0,
)

EXAMPLE_CLEANED_RECORD = CleanedRecord(
    record_id="cln_0001",
    transcript_id="tr_0001",
    cleaned_text="اهلا وسهلا بكم في الحلقة الاولى",
    normalized_text="اهلا وسهلا بكم في الحلقة الاولى",
)

EXAMPLE_PREPARED_DATASET_SAMPLE = PreparedDatasetSample(
    sample_id="sample_0001",
    qa_id="qa_0001",
    transcript_id="tr_0001",
    input_text="سؤال: ما موضوع الحلقة؟\nسياق: اهلا وسهلا بكم في الحلقة الاولى",
    target_text="الحلقة مقدمة عن المشروع",
    split="train",
)
