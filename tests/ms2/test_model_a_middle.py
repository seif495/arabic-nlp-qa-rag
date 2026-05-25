from __future__ import annotations

import unittest

import pytest

tf = pytest.importorskip("tensorflow")

from src.ms2.models.rnn.align import BranchAligner
from src.ms2.models.rnn.film_generator import FiLMGenerator
from src.ms2.models.rnn.gated_merge import GatedMerge
from src.ms2.models.rnn.refinement import RefinementBiGRU


class TestModelAMiddle(unittest.TestCase):
    def test_alignment_uses_per_example_offsets(self) -> None:
        b1 = tf.ones((2, 192))
        b2 = tf.ones((2, 4, 192)) * 2
        positions = tf.reshape(tf.range(20, dtype=tf.float32), (1, 20, 1))
        b3 = tf.tile(positions, (2, 1, 192))
        aligned = BranchAligner()(b1, b2, b3, tf.constant([2, 5]), context_length=4)
        self.assertEqual(aligned[2][0, 0, 0].numpy(), 4.0)
        self.assertEqual(aligned[2][1, 0, 0].numpy(), 7.0)

    def test_gated_merge_sums_to_one(self) -> None:
        branches = tuple(tf.random.normal((2, 8, 192)) for _ in range(3))
        fused, gates = GatedMerge()(branches)
        self.assertEqual(tuple(fused.shape), (2, 8, 192))
        self.assertTrue(
            tf.reduce_all(tf.abs(tf.reduce_sum(gates, axis=-1) - 1.0) < 1e-5).numpy()
        )

    def test_mean_merge_hard_codes_uniform_gates(self) -> None:
        branches = tuple(tf.random.normal((2, 8, 192)) for _ in range(3))
        _, gates = GatedMerge(mean_merge=True)(branches)
        self.assertTrue(tf.reduce_all(tf.abs(gates - (1 / 3)) < 1e-6).numpy())

    def test_refinement_and_film_shapes_identity_init(self) -> None:
        fused = tf.random.normal((2, 384, 192))
        encoder_output, summary = RefinementBiGRU()(fused)
        self.assertEqual(tuple(encoder_output.shape), (2, 384, 192))
        self.assertEqual(tuple(summary.shape), (2, 192))
        gamma, beta = FiLMGenerator()(tf.random.normal((100, 192)))
        self.assertEqual(tuple(gamma.shape), (100, 256))
        self.assertEqual(tuple(beta.shape), (100, 256))
        self.assertLess(float(tf.reduce_max(tf.abs(gamma - 1.0))), 0.01)
        self.assertLess(float(tf.reduce_max(tf.abs(beta))), 0.01)


if __name__ == "__main__":
    unittest.main()
