from __future__ import annotations

import tensorflow as tf

from src.ms2.models.rnn.embedding import SharedTokenEmbedding

D_B1 = 96
D_POOL = 64


class StructuredPooling(tf.keras.layers.Layer):
    """Learned single-sequence pooling with masked softmax, ADR §2.4."""

    def __init__(self, d_pool: int = D_POOL, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.proj = tf.keras.layers.Dense(d_pool, activation="tanh", name="pool_proj")
        self.score = tf.keras.layers.Dense(1, use_bias=False, name="pool_score")
        self.last_weights: tf.Tensor | None = None

    def call(self, sequence: tf.Tensor, mask: tf.Tensor | None = None) -> tf.Tensor:
        scores = tf.squeeze(self.score(self.proj(sequence)), axis=-1)
        if mask is not None:
            scores = tf.where(
                tf.cast(mask, tf.bool),
                scores,
                tf.fill(tf.shape(scores), tf.constant(-1e9, scores.dtype)),
            )
        weights = tf.nn.softmax(tf.cast(scores, tf.float32), axis=-1)
        weights = tf.cast(weights, sequence.dtype)
        self.last_weights = weights
        return tf.reduce_sum(sequence * weights[..., None], axis=1)


class Branch1QuestionEncoder(tf.keras.layers.Layer):
    """Question Bi-LSTM plus structured pooling, ADR §2.4."""

    def __init__(
        self,
        embedding: SharedTokenEmbedding,
        dropout_emb: float = 0.2,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding = embedding
        self.dropout = tf.keras.layers.Dropout(dropout_emb)
        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(D_B1, return_sequences=True), name="branch1_bilstm"
        )
        self.pool = StructuredPooling(name="branch1_structured_pool")

    def call(
        self, question_ids: tf.Tensor, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        mask = tf.not_equal(question_ids, 0)
        x = self.dropout(self.embedding(question_ids), training=training)
        seq = self.encoder(x, mask=mask, training=training)
        pooled = self.pool(seq, mask=mask)
        return pooled, mask
