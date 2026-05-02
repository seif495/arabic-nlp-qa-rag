from __future__ import annotations

import tensorflow as tf

VOCAB_SIZE = 4096
D_TOK = 128
D_CHAR = 32


class SharedTokenEmbedding(tf.keras.layers.Layer):
    """Model A shared BPE embedding initialized per ADR §2.3."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_tok: int = D_TOK, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            d_tok,
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=d_tok**-0.5),
            mask_zero=True,
            name="shared_bpe_embedding",
        )

    @property
    def embeddings(self) -> tf.Variable:
        return self.embedding.embeddings

    def call(self, token_ids: tf.Tensor) -> tf.Tensor:
        return self.embedding(token_ids)

    def compute_mask(self, token_ids: tf.Tensor, mask: tf.Tensor | None = None) -> tf.Tensor:
        del mask
        return tf.not_equal(token_ids, 0)


class CharacterEmbedding(tf.keras.layers.Layer):
    """Model A character embedding for Branch 3 char-CNN, ADR §2.6."""

    def __init__(self, vocab_size: int = 192, d_char: int = D_CHAR, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_char, mask_zero=True, name="char_embedding")

    def call(self, char_ids: tf.Tensor) -> tf.Tensor:
        return self.embedding(char_ids)
