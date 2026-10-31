from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from ms3web.seq2seq import BOS_ID, EOS_ID, SEP_ID, RoPE, Seq2SeqTransformer  # noqa: E402
from src.ms2.models.transformer.rope import RoPE as TFRoPE  # noqa: E402
from src.ms3.models.ms2_web_export import build_model, weight_dict  # noqa: E402
from src.ms3.models.ms2_web_selection import build_candidate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", params=("model_b", "transformer"))
def models(request):
    tf.keras.utils.set_random_seed(7)
    tf_model = build_model(request.param, repo_root=REPO_ROOT)
    _, forward = build_candidate(request.param, REPO_ROOT)
    np_model = Seq2SeqTransformer(weight_dict(request.param, tf_model, dtype="float32"))
    return request.param, tf_model, forward, np_model


def _batch(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    enc = np.zeros((2, 40), dtype=np.int32)
    enc[0, :40] = [BOS_ID, *rng.integers(5, 4096, 30), SEP_ID, *rng.integers(5, 4096, 7), EOS_ID]
    enc[1, :20] = [BOS_ID, *rng.integers(5, 4096, 10), SEP_ID, *rng.integers(5, 4096, 7), EOS_ID]
    dec = np.zeros((2, 12), dtype=np.int32)
    dec[0, :12] = [BOS_ID, *rng.integers(5, 4096, 11)]
    dec[1, :6] = [BOS_ID, *rng.integers(5, 4096, 5)]
    return enc, dec


def test_rope_tables_match_tensorflow() -> None:
    x = np.random.default_rng(0).normal(size=(2, 4, 9, 48)).astype(np.float32)
    expected = TFRoPE(max_length=64)(tf.constant(x), start=5).numpy()
    np.testing.assert_allclose(RoPE(64, 48, 10000.0)(x, start=5), expected, atol=1e-5)


def test_positional_mode_matches_architecture(models) -> None:
    name, _, _, np_model = models
    assert np_model.positional_mode == {"model_b": "rope", "transformer": "sinusoidal"}[name]


def test_teacher_forced_logits_match_tensorflow(models) -> None:
    _, tf_model, forward, np_model = models
    enc, dec = _batch(np.random.default_rng(1))
    expected = forward(
        tf_model,
        {"encoder_input_ids": tf.constant(enc), "decoder_input_ids": tf.constant(dec)},
        False,
    ).numpy()
    actual = np_model.logits(enc, dec)
    # Compare non-padding decoder positions only.
    for row, length in ((0, 12), (1, 6)):
        np.testing.assert_allclose(actual[row, :length], expected[row, :length], atol=2e-3, rtol=1e-3)


def test_kv_cached_greedy_matches_full_recompute(models) -> None:
    _, _, _, np_model = models
    enc, _ = _batch(np.random.default_rng(3))
    result = np_model.greedy_decode(list(enc[0]), max_length=8, max_repeat=None)
    tokens = [BOS_ID]
    for _ in range(8):
        next_id = int(np.argmax(np_model.logits(enc[:1], np.asarray([tokens]))[0, -1]))
        if next_id == EOS_ID:
            break
        tokens.append(next_id)
    assert result.token_ids == tokens[1:]
