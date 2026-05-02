from __future__ import annotations

import unittest

import pytest

tf = pytest.importorskip("tensorflow")

from src.ms2.models.transformer.kv_cache import KVCache
from src.ms2.models.transformer.model_b import ModelB, parameter_audit


class TestModelBFull(unittest.TestCase):
    def test_forward_and_parameter_budget(self) -> None:
        model = ModelB()
        outputs = model(
            (tf.ones((2, 420), dtype=tf.int32), tf.ones((2, 8), dtype=tf.int32))
        )
        self.assertEqual(tuple(outputs["logits"].shape), (2, 8, 4096))
        self.assertEqual(len(outputs["encoder_self_attn_weights"]), 3)
        self.assertEqual(len(outputs["decoder_cross_attn_weights"]), 3)
        self.assertTrue(parameter_audit(model)["within_budget"])

    def test_cache_supports_step_decoding(self) -> None:
        model = ModelB()
        cache = model.init_cache()
        encoder_ids = tf.ones((1, 12), dtype=tf.int32)
        _ = model((encoder_ids, tf.ones((1, 1), dtype=tf.int32)), cache=cache)
        self.assertGreater(cache.populated_steps(), 0)
        self.assertIsInstance(cache, KVCache)

    def test_tiny_loss_decreases_sanity(self) -> None:
        model = ModelB()
        enc = tf.ones((1, 8), dtype=tf.int32)
        dec = tf.ones((1, 3), dtype=tf.int32)
        targets = tf.ones((1, 3), dtype=tf.int32)
        opt = tf.keras.optimizers.Adam(1e-3)
        losses = []
        for _ in range(3):
            with tf.GradientTape() as tape:
                logits = model((enc, dec), training=True)["logits"]
                loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        targets, logits, from_logits=True
                    )
                )
            opt.apply_gradients(
                zip(
                    tape.gradient(loss, model.trainable_variables),
                    model.trainable_variables,
                    strict=False,
                )
            )
            losses.append(float(loss))
        self.assertLessEqual(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
