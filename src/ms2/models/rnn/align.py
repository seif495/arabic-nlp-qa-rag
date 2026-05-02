from __future__ import annotations

import tensorflow as tf


class BranchAligner(tf.keras.layers.Layer):
    """Align Model A branches to the context axis, ADR §2.7."""

    def call(
        self,
        b1_pooled: tf.Tensor,
        b2_seq: tf.Tensor,
        b3_seq: tf.Tensor,
        question_lengths: tf.Tensor,
        context_length: int | tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        l_c = tf.shape(b2_seq)[1] if context_length is None else tf.cast(context_length, tf.int32)
        b1_aligned = tf.tile(b1_pooled[:, None, :], (1, l_c, 1))
        starts = tf.cast(question_lengths, tf.int32) + 2
        offsets = starts[:, None] + tf.range(l_c, dtype=tf.int32)[None, :]
        batch_ids = tf.range(tf.shape(b3_seq)[0], dtype=tf.int32)[:, None]
        batch_ids = tf.tile(batch_ids, (1, l_c))
        gather_idx = tf.stack([batch_ids, offsets], axis=-1)
        b3_aligned = tf.gather_nd(b3_seq, gather_idx)
        return b1_aligned, b2_seq, b3_aligned
