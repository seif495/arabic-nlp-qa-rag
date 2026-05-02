from __future__ import annotations

import unittest

import pytest

tf = pytest.importorskip("tensorflow")

from src.ms2.models.transformer.rope import D_HEAD, RoPE, apply_rope, rope_frequencies


class TestRoPE(unittest.TestCase):
    def test_identity(self) -> None:
        x = tf.random.normal((2, 4, 8, D_HEAD))
        cos = tf.ones((1, 1, 8, D_HEAD // 2))
        sin = tf.zeros((1, 1, 8, D_HEAD // 2))
        self.assertTrue(tf.reduce_all(tf.abs(apply_rope(x, cos, sin) - x) < 1e-6).numpy())

    def test_norm_preservation_and_leading_dims(self) -> None:
        rope = RoPE(max_length=16)
        x = tf.random.normal((3, 2, 4, 5, D_HEAD))
        y = rope(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertLess(float(tf.reduce_max(tf.abs(tf.norm(x, axis=-1) - tf.norm(y, axis=-1)))), 1e-5)

    def test_frequency_table(self) -> None:
        freqs = rope_frequencies().numpy()
        self.assertAlmostEqual(float(freqs[0]), 1.0, places=6)
        self.assertAlmostEqual(float(freqs[-1]), 10000 ** (-46 / 48), places=8)

    def test_relative_position_invariance(self) -> None:
        rope = RoPE(max_length=32)
        q = tf.random.normal((1, 1, 1, D_HEAD))
        k = tf.random.normal((1, 1, 1, D_HEAD))
        m, n, delta = 3, 9, 5
        q1 = apply_rope(q, *rope.tables(1, start=m))
        k1 = apply_rope(k, *rope.tables(1, start=n))
        q2 = apply_rope(q, *rope.tables(1, start=m + delta))
        k2 = apply_rope(k, *rope.tables(1, start=n + delta))
        self.assertAlmostEqual(float(tf.reduce_sum(q1 * k1)), float(tf.reduce_sum(q2 * k2)), places=5)


if __name__ == "__main__":
    unittest.main()
