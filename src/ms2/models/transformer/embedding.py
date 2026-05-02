from __future__ import annotations

import math

import tensorflow as tf

VOCAB_SIZE = 4096
D_MODEL = 192


class TransformerTokenEmbedding(tf.keras.layers.Layer):
    """Shared encoder/decoder embedding scaled by sqrt(d_model), ADR §3.3."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = D_MODEL, dropout: float = 0.1, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            d_model,
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=d_model**-0.5),
            mask_zero=True,
            name="transformer_shared_embedding",
        )
        self.dropout = tf.keras.layers.Dropout(dropout)

    @property
    def embeddings(self) -> tf.Variable:
        return self.embedding.embeddings

    def call(self, token_ids: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.embedding(token_ids) * tf.cast(math.sqrt(self.d_model), self.embedding(token_ids).dtype)
        return self.dropout(x, training=training)


def sinusoidal_encoding(length: int | tf.Tensor, d_model: int = D_MODEL) -> tf.Tensor:
    positions = tf.cast(tf.range(length)[:, None], tf.float32)
    dims = tf.cast(tf.range(d_model)[None, :], tf.float32)
    rates = tf.pow(10000.0, -2.0 * tf.floor(dims / 2.0) / float(d_model))
    angles = positions * rates
    return tf.where(tf.cast(tf.range(d_model) % 2, tf.bool)[None, :], tf.cos(angles), tf.sin(angles))
