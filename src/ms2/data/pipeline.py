from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from src.common.paths import resolve_ms2_paths
from src.ms2.data.length_caps import LENGTH_CAPS
from src.ms2.data.records import MS2DatasetRecord
from src.ms2.data.tokenizer import BOS_ID, EOS_ID, PAD_ID, SEP_ID, MS2Tokenizer


TargetModel = Literal["a", "b"]
BUCKET_BOUNDARIES = (128, 192, 256, 320, 420)
TARGET_TOKENS_PER_BATCH = 16_384
SHUFFLE_BUFFER = 4096
JITTER_FRACTION = 0.2
L_CHAR_MAX = 16


@dataclass(frozen=True)
class PipelineExample:
    sample_id: str
    split: str
    target_model: TargetModel
    encoder_input_ids: list[int]
    decoder_input_ids: list[int]
    decoder_target_ids: list[int]
    loss_mask: list[bool]
    char_matrix: list[list[int]] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_pipeline_example(
    record: MS2DatasetRecord,
    tokenizer: MS2Tokenizer,
    target_model: TargetModel = "a",
    training: bool = False,
    rng: random.Random | None = None,
) -> PipelineExample:
    if target_model not in ("a", "b"):
        raise ValueError("target_model must be 'a' or 'b'")
    context_text = select_context_window(
        record=record,
        tokenizer=tokenizer,
        training=training,
        rng=rng,
    )
    question_ids = tokenizer.encode(record.question_text)[: LENGTH_CAPS["l_q"]]
    context_ids = tokenizer.encode(context_text)[: LENGTH_CAPS["l_c"]]
    answer_ids = tokenizer.encode(record.answer_text)[: max(0, LENGTH_CAPS["l_dec"] - 1)]
    encoder = _pad([BOS_ID, *question_ids, SEP_ID, *context_ids, EOS_ID], LENGTH_CAPS["l_enc"])
    decoder_input = _pad([BOS_ID, *answer_ids], LENGTH_CAPS["l_dec"])
    decoder_target = _pad([*answer_ids, EOS_ID], LENGTH_CAPS["l_dec"])
    char_matrix = None
    if target_model == "a":
        pieces = _encoder_pieces(record.question_text, context_text, tokenizer)
        char_matrix = _pad_char_matrix(pieces, tokenizer, LENGTH_CAPS["l_enc"])
    return PipelineExample(
        sample_id=record.sample_id,
        split=record.split,
        target_model=target_model,
        encoder_input_ids=encoder,
        decoder_input_ids=decoder_input,
        decoder_target_ids=decoder_target,
        loss_mask=[token_id != PAD_ID for token_id in decoder_target],
        char_matrix=char_matrix,
    )


def select_context_window(
    record: MS2DatasetRecord,
    tokenizer: MS2Tokenizer,
    training: bool,
    rng: random.Random | None = None,
) -> str:
    tokens = tokenizer.encode(record.normalized_context)
    if len(tokens) <= LENGTH_CAPS["l_c"]:
        return record.normalized_context
    answer_tokens = tokenizer.encode(record.answer_text)
    start = _find_subsequence(tokens, answer_tokens)
    center = start + len(answer_tokens) // 2 if start >= 0 else len(tokens) // 2
    jitter = 0
    if training:
        generator = rng or random.Random()
        radius = int(LENGTH_CAPS["l_c"] * JITTER_FRACTION)
        jitter = generator.randint(-radius, radius)
    window_start = max(0, min(center + jitter - LENGTH_CAPS["l_c"] // 2, len(tokens) - LENGTH_CAPS["l_c"]))
    return _bounded_token_window_text(
        text=record.normalized_context,
        spans=tokenizer.token_spans(record.normalized_context),
        token_start=window_start,
        token_end=window_start + LENGTH_CAPS["l_c"],
        tokenizer=tokenizer,
    )


def inference_sliding_windows(text: str, tokenizer: MS2Tokenizer) -> list[str]:
    token_ids = tokenizer.encode(text)
    cap = LENGTH_CAPS["l_c"]
    if len(token_ids) <= cap:
        return [text]
    stride = cap // 2
    windows: list[str] = []
    for start in range(0, len(token_ids), stride):
        window_ids = token_ids[start : start + cap]
        if not window_ids:
            break
        windows.append(
            _bounded_token_window_text(
                text=text,
                spans=tokenizer.token_spans(text),
                token_start=start,
                token_end=start + cap,
                tokenizer=tokenizer,
            )
        )
        if start + cap >= len(token_ids):
            break
    return windows


def bucket_batch_sizes(
    boundaries: tuple[int, ...] = BUCKET_BOUNDARIES,
    target_tokens_per_batch: int = TARGET_TOKENS_PER_BATCH,
) -> dict[int, int]:
    return {boundary: max(1, target_tokens_per_batch // boundary) for boundary in boundaries}


def write_pipeline_cache(
    records: list[MS2DatasetRecord],
    tokenizer: MS2Tokenizer,
    repo_root: Path | None = None,
    target_model: TargetModel = "a",
    split: str = "train",
) -> Path:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True, split=f"{split}_{target_model}")
    cache_path = paths.tfrecord_shard_dir / "examples.jsonl"
    if cache_path.exists():
        return cache_path
    selected = [record for record in records if record.split == split]
    rng = random.Random(13)
    with cache_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            example = build_pipeline_example(
                record,
                tokenizer,
                target_model=target_model,
                training=split == "train",
                rng=rng,
            )
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    return cache_path


def _encoder_pieces(question_text: str, context_text: str, tokenizer: MS2Tokenizer) -> list[str]:
    ids = [BOS_ID, *tokenizer.encode(question_text), SEP_ID, *tokenizer.encode(context_text), EOS_ID]
    pieces = [tokenizer.id_to_piece[token_id] if 0 <= token_id < len(tokenizer.id_to_piece) else "<unk>" for token_id in ids]
    return pieces[: LENGTH_CAPS["l_enc"]]


def _pad(values: list[int], target_length: int) -> list[int]:
    return values[:target_length] + [PAD_ID] * max(0, target_length - len(values))


def _pad_char_matrix(
    pieces: list[str],
    tokenizer: MS2Tokenizer,
    target_length: int,
) -> list[list[int]]:
    rows = [tokenizer.encode_chars(piece, max_chars=L_CHAR_MAX) for piece in pieces[:target_length]]
    pad_row = [tokenizer.char_to_id["<pad>"]] * L_CHAR_MAX
    return rows + [pad_row] * max(0, target_length - len(rows))


def _find_subsequence(values: list[int], needle: list[int]) -> int:
    if not needle:
        return -1
    for index in range(0, len(values) - len(needle) + 1):
        if values[index : index + len(needle)] == needle:
            return index
    return -1


def _slice_text_by_token_window(
    text: str,
    spans: list[tuple[int, int]],
    token_start: int,
    token_end: int,
) -> str:
    if not spans:
        return text
    start_index = max(0, min(token_start, len(spans) - 1))
    end_index = max(start_index + 1, min(token_end, len(spans)))
    char_start = spans[start_index][0]
    char_end = spans[end_index - 1][1]
    return text[char_start:char_end]


def _bounded_token_window_text(
    text: str,
    spans: list[tuple[int, int]],
    token_start: int,
    token_end: int,
    tokenizer: MS2Tokenizer,
) -> str:
    window = _slice_text_by_token_window(text, spans, token_start, token_end)
    while len(tokenizer.encode(window)) > LENGTH_CAPS["l_c"] and token_end > token_start + 1:
        token_end -= 1
        window = _slice_text_by_token_window(text, spans, token_start, token_end)
    return window
