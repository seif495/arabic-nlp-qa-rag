from __future__ import annotations

import json
from pathlib import Path

from src.common.paths import resolve_ms2_paths
from src.ms2.data.records import MS2DatasetRecord
from src.ms2.data.tokenizer import (
    VOCAB_SIZE,
    ensure_default_tokenizer,
    train_tokenizer_assets,
    verify_char_coverage,
    verify_special_token_ids,
)


def _records() -> list[MS2DatasetRecord]:
    return [
        MS2DatasetRecord(
            sample_id="s1",
            transcript_id="t1",
            qa_id="q1",
            normalized_context="اهلا العالم 123",
            question_text="QUESTION_SHOULD_NOT_TRAIN_TOKENIZER",
            answer_text="ANSWER_SHOULD_NOT_TRAIN_TOKENIZER",
            split="train",
        )
    ]


def test_tokenizer_assets_have_fixed_vocab_and_special_ids(tmp_path: Path) -> None:
    tokenizer = train_tokenizer_assets(_records(), repo_root=tmp_path)

    assert len(tokenizer.id_to_piece) == VOCAB_SIZE
    assert verify_special_token_ids(tokenizer)
    assert tokenizer.piece_to_id["<pad>"] == 0


def test_tokenizer_training_is_deterministic_and_transcript_only(tmp_path: Path) -> None:
    paths = resolve_ms2_paths(repo_root=tmp_path, create_dirs=True)
    first_tokenizer = train_tokenizer_assets(_records(), repo_root=tmp_path)
    train_tokenizer_assets(_records(), repo_root=tmp_path)
    second_tokenizer = train_tokenizer_assets(_records(), repo_root=tmp_path)

    assert first_tokenizer.id_to_piece == second_tokenizer.id_to_piece
    first_model = paths.tokenizer_path.read_bytes()
    corpus = json.loads((paths.data_processed_ms2 / "ms2_tokenizer_training_corpus_v001.json").read_text(encoding="utf-8"))
    assert corpus == ["اهلا العالم 123"]
    assert b"QUESTION_SHOULD_NOT_TRAIN_TOKENIZER" not in first_model
    assert b"ANSWER_SHOULD_NOT_TRAIN_TOKENIZER" not in first_model


def test_tokenizer_round_trip_and_char_coverage(tmp_path: Path) -> None:
    records = _records()
    tokenizer = train_tokenizer_assets(records, repo_root=tmp_path)

    encoded = tokenizer.encode("اهلا العالم")
    assert tokenizer.decode(encoded) == "اهلا العالم"
    assert verify_char_coverage([records[0].normalized_context], tokenizer)
    assert tokenizer.encode_chars("اهلاالعالم", max_chars=4) == [
        tokenizer.char_to_id[ch] for ch in "اهلا"
    ]
    assert tokenizer.encode_chars("ا", max_chars=3)[1:] == [0, 0]


def test_default_tokenizer_cache_is_scoped_by_repo_root(tmp_path: Path) -> None:
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir(parents=True)
    root_b.mkdir(parents=True)
    records_a = [
        MS2DatasetRecord(
            sample_id="sa",
            transcript_id="ta",
            qa_id="qa",
            normalized_context="alpha_only",
            question_text="q",
            answer_text="a",
            split="train",
        )
    ]
    records_b = [
        MS2DatasetRecord(
            sample_id="sb",
            transcript_id="tb",
            qa_id="qb",
            normalized_context="beta_only",
            question_text="q",
            answer_text="a",
            split="train",
        )
    ]
    train_tokenizer_assets(records_a, repo_root=root_a)
    train_tokenizer_assets(records_b, repo_root=root_b)

    tokenizer_a = ensure_default_tokenizer(repo_root=root_a)
    tokenizer_b = ensure_default_tokenizer(repo_root=root_b)

    assert tokenizer_a.processor.serialized_model_proto() != tokenizer_b.processor.serialized_model_proto()
