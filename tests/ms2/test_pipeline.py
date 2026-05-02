from __future__ import annotations

import json
import random
from pathlib import Path

from src.ms2.data.length_caps import LENGTH_CAPS
from src.ms2.data.pipeline import (
    BUCKET_BOUNDARIES,
    TARGET_TOKENS_PER_BATCH,
    bucket_batch_sizes,
    build_pipeline_example,
    inference_sliding_windows,
    select_context_window,
    write_pipeline_cache,
)
from src.ms2.data.records import MS2DatasetRecord
from src.ms2.data.tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SEP_ID,
    train_tokenizer_assets,
)


def _record(
    context: str, answer: str = "tok5", split: str = "train"
) -> MS2DatasetRecord:
    return MS2DatasetRecord(
        sample_id="s1",
        transcript_id="t1",
        qa_id="q1",
        normalized_context=context,
        question_text="tok1 tok2",
        answer_text=answer,
        split=split,
    )


def test_pipeline_formats_sequences_and_model_schemas(tmp_path: Path) -> None:
    record = _record("tok1 tok2 tok3 tok4 tok5", answer="tok5")
    tokenizer = train_tokenizer_assets([record], repo_root=tmp_path)

    model_a = build_pipeline_example(record, tokenizer, target_model="a")
    model_b = build_pipeline_example(record, tokenizer, target_model="b")

    assert model_a.encoder_input_ids[0] == BOS_ID
    assert SEP_ID in model_a.encoder_input_ids
    assert model_a.decoder_input_ids[0] == BOS_ID
    assert EOS_ID in model_a.decoder_target_ids
    assert len(model_a.encoder_input_ids) == LENGTH_CAPS["l_enc"]
    assert len(model_a.decoder_input_ids) == LENGTH_CAPS["l_dec"]
    assert len(model_a.decoder_target_ids) == LENGTH_CAPS["l_dec"]
    assert model_a.loss_mask == [
        token_id != PAD_ID for token_id in model_a.decoder_target_ids
    ]
    assert model_a.char_matrix is not None
    assert len(model_a.char_matrix) == LENGTH_CAPS["l_enc"]
    assert len(model_a.char_matrix[0]) == 16
    assert model_b.char_matrix is None


def test_training_context_jitter_is_bounded_and_centered(tmp_path: Path) -> None:
    context = " ".join(f"tok{i}" for i in range(900))
    record = _record(context, answer="tok500")
    tokenizer = train_tokenizer_assets([record], repo_root=tmp_path)
    windows = []

    for seed in range(200):
        window = select_context_window(
            record, tokenizer, training=True, rng=random.Random(seed)
        )
        windows.append(window)

    assert all("tok500" in window for window in windows)
    assert all(
        len(tokenizer.encode(window)) <= LENGTH_CAPS["l_c"] for window in windows
    )
    assert len(set(windows)) > 1


def test_context_window_preserves_original_text_slicing(tmp_path: Path) -> None:
    context = ",".join(f"tok{i}" for i in range(900))
    record = _record(context, answer="tok500")
    tokenizer = train_tokenizer_assets([record], repo_root=tmp_path)

    window = select_context_window(record, tokenizer, training=False)

    assert "," in window
    assert " , " not in window


def test_inference_windows_bucket_sizes_and_cache_reuse(tmp_path: Path) -> None:
    context = " ".join(f"tok{i}" for i in range(900))
    record = _record(context)
    tokenizer = train_tokenizer_assets([record], repo_root=tmp_path)

    windows = inference_sliding_windows(context, tokenizer)
    assert len(windows) > 1
    assert all(
        len(tokenizer.encode(window)) <= LENGTH_CAPS["l_c"] for window in windows
    )
    assert windows[0] != windows[1]

    batch_sizes = bucket_batch_sizes()
    assert tuple(batch_sizes) == BUCKET_BOUNDARIES
    for boundary, batch_size in batch_sizes.items():
        assert abs((boundary * batch_size) - TARGET_TOKENS_PER_BATCH) <= boundary

    cache_path = write_pipeline_cache(
        [record], tokenizer, repo_root=tmp_path, target_model="a"
    )
    cache_path.write_text("sentinel\n", encoding="utf-8")
    assert (
        write_pipeline_cache([record], tokenizer, repo_root=tmp_path, target_model="a")
        == cache_path
    )
    assert cache_path.read_text(encoding="utf-8") == "sentinel\n"


def test_cache_training_jitter_uses_progressive_rng_state(tmp_path: Path) -> None:
    context = " ".join(f"tok{i}" for i in range(900))
    records = [
        _record(context, answer="tok500", split="train"),
        MS2DatasetRecord(
            sample_id="s2",
            transcript_id="t2",
            qa_id="q2",
            normalized_context=context,
            question_text="tok1 tok2",
            answer_text="tok500",
            split="train",
        ),
    ]
    tokenizer = train_tokenizer_assets(records, repo_root=tmp_path)

    cache_path = write_pipeline_cache(
        records, tokenizer, repo_root=tmp_path, target_model="a"
    )
    lines = [
        json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines()
    ]
    first_context_start = lines[0]["encoder_input_ids"].index(SEP_ID) + 1
    second_context_start = lines[1]["encoder_input_ids"].index(SEP_ID) + 1

    assert first_context_start == second_context_start
    assert lines[0]["encoder_input_ids"] != lines[1]["encoder_input_ids"]
