from __future__ import annotations

import tensorflow as tf

from src.ms2.models.rnn.embedding import D_CHAR

D_CHARCNN = 64
KERNEL_SIZES = (2, 3, 4)


class CharCNN(tf.keras.layers.Layer):
    """Per-token character CNN matching ADR §2.6 reshape/conv/pool contract."""

    def __init__(self, d_charcnn: int = D_CHARCNN, **kwargs: object) -> None:
        super().__init__(**kwargs)
        base_filters = d_charcnn // len(KERNEL_SIZES)
        filters = [base_filters, base_filters, d_charcnn - 2 * base_filters]
        self.convs = [
            tf.keras.layers.Conv1D(
                filters=count,
                kernel_size=kernel,
                activation="relu",
                padding="valid",
                name=f"char_conv_k{kernel}",
            )
            for kernel, count in zip(KERNEL_SIZES, filters, strict=True)
        ]
        self.pool = tf.keras.layers.GlobalMaxPool1D(name="char_global_max_pool")

    def call(self, char_emb: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(char_emb)
        batch_size, seq_len, char_len = shape[0], shape[1], shape[2]
        flat = tf.reshape(char_emb, (batch_size * seq_len, char_len, D_CHAR))
        pooled = [self.pool(conv(flat)) for conv in self.convs]
        features = tf.concat(pooled, axis=-1)
        return tf.reshape(features, (batch_size, seq_len, D_CHARCNN))
