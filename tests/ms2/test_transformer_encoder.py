from __future__ import annotations

import unittest

import pytest

tf = pytest.importorskip("tensorflow")

from src.ms2.models.transformer.embedding import TransformerTokenEmbedding
from src.ms2.models.transformer.encoder import TransformerEncoder


class TestTransformerEncoder(unittest.TestCase):
    def test_encoder_output_shape(self) -> None:
        emb = TransformerTokenEmbedding()
        enc = TransformerEncoder(emb)
        output, attentions, mask = enc(tf.ones((2, 420), dtype=tf.int32))
        self.assertEqual(tuple(output.shape), (2, 420, 192))
        self.assertEqual(len(attentions), 3)
        self.assertEqual(tuple(mask.shape), (2, 420))

    def test_sinusoidal_and_no_pe_modes(self) -> None:
        emb = TransformerTokenEmbedding()
        self.assertEqual(TransformerEncoder(emb, positional_mode="sinusoidal_pe").positional_mode, "sinusoidal_pe")
        self.assertEqual(TransformerEncoder(emb, positional_mode="no_pe").positional_mode, "no_pe")


if __name__ == "__main__":
    unittest.main()
