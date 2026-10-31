"""The MS2 models the website runs, behind one small interface.

``python -m src.cli.ms3 export-web`` writes ``assets/ms2_models.json``: the
exported MS2 models ranked by ``select-ms2-web`` (dev token-F1), the default
being the winner. :func:`load_runtimes` returns one runtime per model, each
exposing:

- ``generate(question, windows)``: greedy answers, one per context window;
- ``question_vector`` / ``context_vectors``: L2-normalised encodings used to
  re-rank retrieved chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from ms3web.rnn import ModelANumpy
from ms3web.seq2seq import BOS_ID, EOS_ID, PAD_ID, SEP_ID, DecodeResult, Seq2SeqTransformer
from ms3web.tokenizer import L_C, L_ENC, L_Q, MS2Tokenizer

MODELS_INDEX_NAME = "ms2_models.json"
TOKENIZER_NAME = "ms2_tokenizer.model"
CHAR_VOCAB_NAME = "ms2_char_vocab.json"
CHAR_PAD, CHAR_UNK, MAX_CHARS = "<pad>", "<unk>", 16


class MS2Runtime(Protocol):
    info: dict
    tokenizer: MS2Tokenizer

    def generate(self, question: str, windows: list[list[int]]) -> list[DecodeResult]: ...

    def question_vector(self, question: str) -> np.ndarray: ...

    def context_vectors(self, contexts: list[str]) -> np.ndarray: ...


def pad(ids: list[int], length: int) -> list[int]:
    ids = ids[:length]
    return ids + [PAD_ID] * (length - len(ids))


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


class TransformerRuntime:
    """``model_b`` (RoPE) or ``transformer`` (sinusoidal) via :class:`Seq2SeqTransformer`."""

    def __init__(self, model: Seq2SeqTransformer, tokenizer: MS2Tokenizer, info: dict) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.info = info

    def generate(self, question: str, windows: list[list[int]]) -> list[DecodeResult]:
        return [self.model.greedy_decode(self.tokenizer.encoder_input(question, w)) for w in windows]

    def question_vector(self, question: str) -> np.ndarray:
        return self.model.pooled_encoding(pad(self.tokenizer.encode(question), L_Q))

    def context_vectors(self, contexts: list[str]) -> np.ndarray:
        return np.stack([self.model.pooled_encoding(pad(self.tokenizer.encode(c), L_C)) for c in contexts])


class ModelARuntime:
    """``model_a`` (tri-encoder RNN + FiLM) via :class:`ModelANumpy`, batched over windows."""

    def __init__(self, model: ModelANumpy, tokenizer: MS2Tokenizer, char_to_id: dict[str, int], info: dict) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.char_to_id = char_to_id
        self.info = info

    def _char_row(self, piece: str) -> list[int]:
        """``MS2Tokenizer.encode_chars`` from src/ms2/data/tokenizer.py."""
        unk = self.char_to_id[CHAR_UNK]
        ids = [self.char_to_id.get(ch, unk) for ch in piece[:MAX_CHARS]]
        return ids + [self.char_to_id[CHAR_PAD]] * (MAX_CHARS - len(ids))

    def _inputs(self, question: str, windows: list[list[int]]) -> tuple[np.ndarray, ...]:
        """Same tensors as ``build_pipeline_example(target_model="a")``."""
        q_ids = self.tokenizer.encode(question)[:L_Q]
        rows_q, rows_c, rows_joint, rows_chars = [], [], [], []
        pad_row = [self.char_to_id[CHAR_PAD]] * MAX_CHARS
        for window in windows:
            c_ids = window[:L_C]
            joint = [BOS_ID, *q_ids, SEP_ID, *c_ids, EOS_ID][:L_ENC]
            pieces = [self.tokenizer.processor.id_to_piece(i) for i in joint]
            chars = [self._char_row(p) for p in pieces]
            rows_q.append(pad(q_ids, L_Q))
            rows_c.append(pad(c_ids, L_C))
            rows_joint.append(pad(joint, L_ENC))
            rows_chars.append(chars + [pad_row] * (L_ENC - len(chars)))
        return tuple(np.asarray(rows) for rows in (rows_q, rows_c, rows_joint, rows_chars))

    def generate(self, question: str, windows: list[list[int]]) -> list[DecodeResult]:
        if not windows:
            return []
        summary = self.model.encode(*self._inputs(question, windows)).summary
        return self.model.greedy_decode(summary)

    def question_vector(self, question: str) -> np.ndarray:
        ids = np.asarray([pad(self.tokenizer.encode(question), L_Q)])
        return _l2(self.model.question_branch(ids)[0])

    def context_vectors(self, contexts: list[str]) -> np.ndarray:
        ids = np.asarray([pad(self.tokenizer.encode(c), L_C) for c in contexts])
        seq, mask = self.model.context_branch(ids)
        weights = mask.astype(np.float32)[..., None]
        pooled = (seq * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1.0)
        return _l2(pooled)


def load_runtimes(assets_dir: Path) -> tuple[dict[str, MS2Runtime], str | None]:
    """Return ``({model_name: runtime}, default_model_name)``; empty if not exported."""
    index_path = assets_dir / MODELS_INDEX_NAME
    tokenizer_path = assets_dir / TOKENIZER_NAME
    if not index_path.exists() or not tokenizer_path.exists():
        return {}, None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    tokenizer = MS2Tokenizer(tokenizer_path)
    char_vocab: dict[str, int] | None = None
    runtimes: dict[str, MS2Runtime] = {}
    for info in index["models"]:
        weights = assets_dir / info["weights"]
        if not weights.exists():
            continue
        if info["runtime"] == "seq2seq_transformer":
            runtimes[info["model"]] = TransformerRuntime(Seq2SeqTransformer.load(weights), tokenizer, info)
        elif info["runtime"] == "model_a_rnn":
            if char_vocab is None:
                char_vocab = json.loads((assets_dir / CHAR_VOCAB_NAME).read_text(encoding="utf-8"))
            runtimes[info["model"]] = ModelARuntime(ModelANumpy.load(weights), tokenizer, char_vocab, info)
        else:
            raise ValueError(f"unknown MS2 web runtime {info['runtime']!r}")
    default = index.get("default")
    if default not in runtimes:
        default = next(iter(runtimes), None)
    return runtimes, default
