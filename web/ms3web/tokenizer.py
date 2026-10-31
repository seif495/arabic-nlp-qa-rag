"""MS2 SentencePiece tokenizer (same model file Model B was trained with)."""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

from ms3web.seq2seq import BOS_ID, EOS_ID, PAD_ID, SEP_ID

SPECIAL_IDS = frozenset({PAD_ID, BOS_ID, EOS_ID, SEP_ID})

# LENGTH_CAPS from src/ms2/data/length_caps.py
L_Q = 32
L_C = 284
L_ENC = 320


class MS2Tokenizer:
    def __init__(self, model_file: str | Path) -> None:
        self.processor = spm.SentencePieceProcessor(model_file=str(model_file))

    def encode(self, text: str) -> list[int]:
        return list(self.processor.encode(text, out_type=int))

    def decode(self, ids: list[int]) -> str:
        return self.processor.decode([i for i in ids if i not in SPECIAL_IDS])

    def encoder_input(self, question: str, context_ids: list[int]) -> list[int]:
        """``[BOS] q [SEP] c [EOS]`` padded to ``L_ENC`` (src/ms2/data/pipeline.py)."""
        q_ids = self.encode(question)[:L_Q]
        ids = [BOS_ID, *q_ids, SEP_ID, *context_ids[:L_C], EOS_ID][:L_ENC]
        return ids + [PAD_ID] * (L_ENC - len(ids))

    def context_windows(self, text: str, max_windows: int = 4) -> list[list[int]]:
        """Stride-``L_C/2`` windows, as in ``src/ms2/inference/windowing.py``."""
        tokens = self.encode(text)
        if len(tokens) <= L_C:
            return [tokens]
        stride = L_C // 2
        windows = []
        for start in range(0, len(tokens), stride):
            windows.append(tokens[start : start + L_C])
            if start + L_C >= len(tokens) or len(windows) == max_windows:
                break
        return windows
