"""The web Model A runtime must build the same tensors as the MS2 training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import sentencepiece as spm

from ms3web.ms2 import ModelARuntime
from ms3web.tokenizer import MS2Tokenizer as WebTokenizer
from src.ms2.data.pipeline import build_pipeline_example
from src.ms2.data.records import MS2DatasetRecord
from src.ms2.data.tokenizer import MS2Tokenizer, _pad_pieces_to_vocab

ASSETS = Path(__file__).resolve().parents[2] / "web/ms3web/assets"


@pytest.mark.skipif(not (ASSETS / "ms2_char_vocab.json").exists(), reason="web assets not exported")
def test_model_a_inputs_match_ms2_pipeline() -> None:
    processor = spm.SentencePieceProcessor(model_file=str(ASSETS / "ms2_tokenizer.model"))
    char_vocab = json.loads((ASSETS / "ms2_char_vocab.json").read_text(encoding="utf-8"))
    pieces = _pad_pieces_to_vocab([processor.id_to_piece(i) for i in range(processor.get_piece_size())])
    src_tokenizer = MS2Tokenizer(pieces, char_vocab, processor)
    question = "مين اللي بنى تاج محل؟"
    context = "شاه جهان بنى تاج محل في أجرا عشان مراته ممتاز محل، وده خد سنين طويلة 😊 Taj Mahal."
    record = MS2DatasetRecord("s", "t", "q", context, question, "شاه جهان", "dev")
    example = build_pipeline_example(record, src_tokenizer, target_model="a", training=False)

    runtime = ModelARuntime(model=None, tokenizer=WebTokenizer(ASSETS / "ms2_tokenizer.model"), char_to_id=char_vocab, info={})
    q, c, joint, chars = runtime._inputs(question, [src_tokenizer.encode(context)])
    np.testing.assert_array_equal(q[0], example.question_ids)
    np.testing.assert_array_equal(c[0], example.context_ids)
    np.testing.assert_array_equal(joint[0], example.encoder_input_ids)
    np.testing.assert_array_equal(chars[0], np.asarray(example.char_matrix))
