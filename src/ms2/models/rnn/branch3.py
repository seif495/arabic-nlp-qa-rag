from __future__ import annotations

import tensorflow as tf

from src.ms2.models.rnn.char_cnn import CharCNN
from src.ms2.models.rnn.embedding import CharacterEmbedding, SharedTokenEmbedding

D_B3 = 96


class Branch3JointEncoder(tf.keras.layers.Layer):
    """Joint token+char Branch 3 encoder, ADR §2.6."""

    def __init__(
        self,
        token_embedding: SharedTokenEmbedding,
        char_vocab_size: int = 192,
        dropout_emb: float = 0.2,
        use_char_cnn: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.token_embedding = token_embedding
        self.use_char_cnn = use_char_cnn
        self.char_embedding = CharacterEmbedding(vocab_size=char_vocab_size)
        self.char_cnn = CharCNN(name="branch3_char_cnn")
        self.dropout = tf.keras.layers.Dropout(dropout_emb)
        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(D_B3, return_sequences=True), name="branch3_bilstm"
        )
        self.plain_projection = tf.keras.layers.Dense(
            192, name="plain_branch3_projection"
        )

    def call(
        self, joint_ids: tf.Tensor, char_ids: tf.Tensor, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        mask = tf.not_equal(joint_ids, 0)
        tok = self.token_embedding(joint_ids)
        if self.use_char_cnn:
            chars = self.char_cnn(self.char_embedding(char_ids))
            x = tf.concat([tok, chars], axis=-1)
        else:
            x = self.plain_projection(tok)
        x = self.dropout(x, training=training)
        return self.encoder(x, mask=mask, training=training), mask
