from __future__ import annotations

import unittest

import pytest

tf = pytest.importorskip("tensorflow")

from src.ms2.models.rnn.decoder import FiLMLSTMDecoder
from src.ms2.models.rnn.embedding import SharedTokenEmbedding
from src.ms2.models.rnn.model_a import ModelA, parameter_audit


class TestModelAFull(unittest.TestCase):
    def test_decoder_shapes_and_identity_film(self) -> None:
        emb = SharedTokenEmbedding()
        decoder = FiLMLSTMDecoder(emb)
        dec_ids = tf.ones((2, 5), dtype=tf.int32)
        z = tf.random.normal((2, 192))
        gamma = tf.ones((2, 256))
        beta = tf.zeros((2, 256))
        logits = decoder(dec_ids, z, gamma, beta)
        self.assertEqual(tuple(logits.shape), (2, 5, 4096))

    def test_model_a_forward_outputs_and_tied_embedding(self) -> None:
        model = ModelA()
        outputs = model(
            (
                tf.ones((2, 32), dtype=tf.int32),
                tf.ones((2, 384), dtype=tf.int32),
                tf.ones((2, 420), dtype=tf.int32),
                tf.ones((2, 420, 16), dtype=tf.int32),
                tf.ones((2, 8), dtype=tf.int32),
            )
        )
        self.assertEqual(tuple(outputs["logits"].shape), (2, 8, 4096))
        self.assertEqual(tuple(outputs["gates"].shape), (2, 384, 3))
        embedding_ref = model.decoder.output_projection.embedding.embeddings
        self.assertIs(embedding_ref, model.shared_embedding.embeddings)
        audit = parameter_audit(model)
        self.assertTrue(audit["within_budget"])

    def test_tiny_loss_decreases_sanity(self) -> None:
        model = ModelA()
        batch = (
            tf.ones((1, 4), dtype=tf.int32),
            tf.ones((1, 8), dtype=tf.int32),
            tf.ones((1, 14), dtype=tf.int32),
            tf.ones((1, 14, 16), dtype=tf.int32),
            tf.ones((1, 3), dtype=tf.int32),
        )
        targets = tf.ones((1, 3), dtype=tf.int32)
        opt = tf.keras.optimizers.Adam(1e-3)
        losses = []
        for _ in range(3):
            with tf.GradientTape() as tape:
                logits = model(batch, training=True)["logits"]
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True))
            opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables, strict=False))
            losses.append(float(loss))
        self.assertLessEqual(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
