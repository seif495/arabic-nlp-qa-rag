from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from ms3web.model_b import BOS_ID, EOS_ID, SEP_ID, ModelBNumpy, RoPE  # noqa: E402
from src.ms2.models.transformer.rope import RoPE as TFRoPE  # noqa: E402
from src.ms3.models.model_b_export import build_model_b, model_b_weight_dict  # noqa: E402


@pytest.fixture(scope="module")
def models():
    tf.keras.utils.set_random_seed(7)
    tf_model = build_model_b()
    np_model = ModelBNumpy(model_b_weight_dict(tf_model, dtype="float32"))
    return tf_model, np_model


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


def test_teacher_forced_logits_match_tensorflow(models) -> None:
    tf_model, np_model = models
    enc, dec = _batch(np.random.default_rng(1))
    expected = tf_model((tf.constant(enc), tf.constant(dec)), training=False)["logits"]
    actual = np_model.logits(enc, dec)
    # Compare non-padding decoder positions only.
    for row, length in ((0, 12), (1, 6)):
        np.testing.assert_allclose(
            actual[row, :length], expected.numpy()[row, :length], atol=2e-3, rtol=1e-3
        )


def test_encoder_states_match_tensorflow(models) -> None:
    tf_model, np_model = models
    enc, _ = _batch(np.random.default_rng(2))
    expected, _, _ = tf_model.encoder(tf.constant(enc), training=False)
    actual, mask = np_model.encode(enc)
    np.testing.assert_allclose(actual[mask], expected.numpy()[mask], atol=1e-3)


def test_kv_cached_greedy_matches_full_recompute(models) -> None:
    _, np_model = models
    enc, _ = _batch(np.random.default_rng(3))
    result = np_model.greedy_decode(list(enc[0]), max_length=8, max_repeat=None)
    tokens = [BOS_ID]
    for _ in range(8):
        next_id = int(np.argmax(np_model.logits(enc[:1], np.asarray([tokens]))[0, -1]))
        if next_id == EOS_ID:
            break
        tokens.append(next_id)
    assert result.token_ids == tokens[1:]
