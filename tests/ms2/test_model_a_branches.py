from __future__ import annotations

import unittest

import pytest

tf = pytest.importorskip("tensorflow")

from src.ms2.models.rnn.branch1 import Branch1QuestionEncoder
from src.ms2.models.rnn.branch2 import Branch2ContextEncoder
from src.ms2.models.rnn.branch3 import Branch3JointEncoder
from src.ms2.models.rnn.char_cnn import KERNEL_SIZES
from src.ms2.models.rnn.embedding import SharedTokenEmbedding


class TestModelABranches(unittest.TestCase):
    def test_branch_shape_contracts(self) -> None:
        emb = SharedTokenEmbedding()
        b1 = Branch1QuestionEncoder(emb)
        b2 = Branch2ContextEncoder(emb)
        b3 = Branch3JointEncoder(emb)
        question = tf.ones((2, 32), dtype=tf.int32)
        context = tf.ones((2, 384), dtype=tf.int32)
        joint = tf.ones((2, 420), dtype=tf.int32)
        chars = tf.ones((2, 420, 16), dtype=tf.int32)
        self.assertEqual(tuple(b1(question)[0].shape), (2, 192))
        self.assertEqual(tuple(b2(context)[0].shape), (2, 384, 192))
        self.assertEqual(tuple(b3(joint, chars)[0].shape), (2, 420, 192))

    def test_structured_pooling_masks_padding(self) -> None:
        emb = SharedTokenEmbedding()
        b1 = Branch1QuestionEncoder(emb)
        question = tf.constant([[5, 6, 0, 0]], dtype=tf.int32)
        _ = b1(question)
        weights = b1.pool.last_weights.numpy()[0]
        self.assertAlmostEqual(float(weights[2]), 0.0, places=6)
        self.assertAlmostEqual(float(weights[3]), 0.0, places=6)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

    def test_char_cnn_kernel_sizes_and_output(self) -> None:
        emb = SharedTokenEmbedding()
        b3 = Branch3JointEncoder(emb)
        self.assertEqual(
            tuple(conv.kernel_size[0] for conv in b3.char_cnn.convs), KERNEL_SIZES
        )
        joint = tf.ones((2, 8), dtype=tf.int32)
        chars = tf.ones((2, 8, 16), dtype=tf.int32)
        out, mask = b3(joint, chars)
        self.assertEqual(tuple(out.shape), (2, 8, 192))
        self.assertEqual(tuple(mask.shape), (2, 8))


if __name__ == "__main__":
    unittest.main()
