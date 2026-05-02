from __future__ import annotations

import tensorflow as tf

from src.ms2.models.rnn.embedding import SharedTokenEmbedding

D_B2 = 96


class Branch2ContextEncoder(tf.keras.layers.Layer):
    """Two-layer context Bi-GRU stack, ADR §2.5."""

    def __init__(self, embedding: SharedTokenEmbedding, dropout_emb: float = 0.2, dropout_rnn: float = 0.3, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.embedding = embedding
        self.dropout_emb = tf.keras.layers.Dropout(dropout_emb)
        self.gru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(D_B2, return_sequences=True), name="branch2_bigru_1")
        self.dropout_rnn = tf.keras.layers.Dropout(dropout_rnn)
        self.gru2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(D_B2, return_sequences=True), name="branch2_bigru_2")

    def call(self, context_ids: tf.Tensor, training: bool = False) -> tuple[tf.Tensor, tf.Tensor]:
        mask = tf.not_equal(context_ids, 0)
        x = self.dropout_emb(self.embedding(context_ids), training=training)
        x = self.gru1(x, mask=mask, training=training)
        x = self.dropout_rnn(x, training=training)
        x = self.gru2(x, mask=mask, training=training)
        return x, mask
