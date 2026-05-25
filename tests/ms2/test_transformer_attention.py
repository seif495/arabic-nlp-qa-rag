from __future__ import annotations

import unittest

import pytest

tf = pytest.importorskip("tensorflow")

from src.ms2.models.transformer.attention import (
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
)


class TestTransformerAttention(unittest.TestCase):
    def test_padding_and_causal_masks(self) -> None:
        attn = MultiHeadSelfAttention()
        x = tf.random.normal((1, 4, 192))
        _, weights = attn(
            x,
            padding_mask=tf.constant([[True, True, False, False]]),
            use_causal_mask=True,
        )
        self.assertLess(float(tf.reduce_max(weights[..., 2:])), 1e-6)
        self.assertLess(float(weights[0, 0, 0, 1]), 1e-6)

    def test_softmax_is_float32_and_rope_contract(self) -> None:
        attn = MultiHeadSelfAttention(use_rope=True)
        _ = attn(tf.random.normal((1, 3, 192)))
        self.assertEqual(attn.softmax_dtype, tf.float32)
        self.assertIsNotNone(attn.rope)
        cross = MultiHeadCrossAttention()
        self.assertIsNone(cross.rope)


if __name__ == "__main__":
    unittest.main()
