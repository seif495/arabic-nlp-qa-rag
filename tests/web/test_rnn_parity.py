from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from ms3web.rnn import ModelANumpy  # noqa: E402
from src.ms2.models.rnn.model_a import ModelA  # noqa: E402
from src.ms3.models.ms2_web_export import model_a_weight_dict  # noqa: E402

L_Q, L_C, CHARS = 8, 20, 16
L_ENC = L_Q + L_C + 3


@pytest.fixture(scope="module")
def models():
    tf.keras.utils.set_random_seed(11)
    tf_model = ModelA(char_vocab_size=60)
    inputs = _inputs(np.random.default_rng(0))
    tf_model(tuple(tf.constant(x) for x in inputs), training=False)
    return tf_model, ModelANumpy(model_a_weight_dict(tf_model, dtype="float32"))


def _inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    q = np.zeros((2, L_Q), np.int32)
    c = np.zeros((2, L_C), np.int32)
    q[0, :6], q[1, :3] = rng.integers(5, 4096, 6), rng.integers(5, 4096, 3)
    c[0, :20], c[1, :11] = rng.integers(5, 4096, 20), rng.integers(5, 4096, 11)
    joint = np.zeros((2, L_ENC), np.int32)
    for row in range(2):
        qs, cs = q[row][q[row] != 0], c[row][c[row] != 0]
        ids = [2, *qs, 4, *cs, 3]
        joint[row, : len(ids)] = ids
    chars = rng.integers(1, 60, (2, L_ENC, CHARS)).astype(np.int32)
    chars[:, :, 10:] = 0
    chars[joint == 0] = 0
    dec = np.zeros((2, 7), np.int32)
    dec[0, :7], dec[1, :4] = [2, *rng.integers(5, 4096, 6)], [2, *rng.integers(5, 4096, 3)]
    return q, c, joint, chars, dec


def test_encoder_summary_and_branches_match_tensorflow(models) -> None:
    tf_model, np_model = models
    q, c, joint, chars, dec = _inputs(np.random.default_rng(1))
    expected = tf_model(tuple(tf.constant(x) for x in (q, c, joint, chars, dec)), training=False)
    actual = np_model.encode(q, c, joint, chars)
    np.testing.assert_allclose(actual.summary, expected["encoder_summary"].numpy(), atol=1e-4)
    b1, _ = tf_model.branch1(tf.constant(q), training=False)
    np.testing.assert_allclose(actual.question_pooled, b1.numpy(), atol=1e-4)


def test_teacher_forced_logits_match_tensorflow(models) -> None:
    tf_model, np_model = models
    q, c, joint, chars, dec = _inputs(np.random.default_rng(2))
    expected = tf_model(tuple(tf.constant(x) for x in (q, c, joint, chars, dec)), training=False)["logits"].numpy()
    actual = np_model.decoder_logits(np_model.encode(q, c, joint, chars).summary, dec)
    for row, length in ((0, 7), (1, 4)):
        np.testing.assert_allclose(actual[row, :length], expected[row, :length], atol=2e-3, rtol=1e-3)


def test_batched_greedy_matches_teacher_forced_argmax(models) -> None:
    _, np_model = models
    q, c, joint, chars, _ = _inputs(np.random.default_rng(3))
    summary = np_model.encode(q, c, joint, chars).summary
    results = np_model.greedy_decode(summary, max_length=6, max_repeat=None)
    for row, result in enumerate(results):
        tokens = [2]
        for _ in range(6):
            nxt = int(np_model.decoder_logits(summary[row : row + 1], np.asarray([tokens]))[0, -1].argmax())
            if nxt == 3:
                break
            tokens.append(nxt)
        assert result.token_ids == tokens[1:]
