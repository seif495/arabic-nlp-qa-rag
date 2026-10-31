from __future__ import annotations

import math

import tensorflow as tf

from src.ms2.util import PipelineStep, get_config_value, load_pipeline_config

VOCAB_SIZE = 4096
D_MODEL = 192


class TransformerTokenEmbedding(tf.keras.layers.Layer):
    """Shared encoder/decoder embedding scaled by sqrt(d_model), ADR §3.3."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        dropout: float = 0.1,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            d_model,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                stddev=d_model**-0.5
            ),
            mask_zero=True,
            name="transformer_shared_embedding",
        )
        self.dropout = tf.keras.layers.Dropout(dropout)

    @property
    def embeddings(self) -> tf.Variable:
        return self.embedding.embeddings

    def call(self, token_ids: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.embedding(token_ids) * tf.cast(
            math.sqrt(self.d_model), self.embedding(token_ids).dtype
        )
        return self.dropout(x, training=training)


def sinusoidal_encoding(length: int | tf.Tensor, d_model: int = D_MODEL) -> tf.Tensor:
    positions = tf.cast(tf.range(length)[:, None], tf.float32)
    dims = tf.cast(tf.range(d_model)[None, :], tf.float32)
    rates = tf.pow(10000.0, -2.0 * tf.floor(dims / 2.0) / float(d_model))
    angles = positions * rates
    return tf.where(
        tf.cast(tf.range(d_model) % 2, tf.bool)[None, :], tf.cos(angles), tf.sin(angles)
    )


# Vanilla sinusoidal Transformer (``model.py``) embedding. Restored from
# e3fcf23 after a later reformat dropped it, which broke ``TransformerModel``.
class TokenEmbedding(tf.keras.layers.Layer):
    """
    Token embedding layer for the vanilla Transformer (Model B).

    This layer maps integer token IDs to dense vectors and then scales the
    embeddings by ``sqrt(d_model)``. The scaling follows the standard
    Transformer convention and keeps activation magnitudes in a stable range.

    Input shape:
        ``(B, L)``

    Output shape:
        ``(B, L, d_model)``
    """

    def __init__(self, vocab_size: int | None = None, **kwargs: object) -> None:
        """
        Initialize the token embedding layer.

        Args:
            vocab_size: Optional explicit vocabulary size override. If omitted,
                vocab size is derived from the preprocessed vocab file.
            **kwargs: Additional keyword arguments passed to
                ``tf.keras.layers.Layer``.

        Returns:
            None.
        """
        ### init ###
        super().__init__(**kwargs)

        ### load model-definition config ###
        config = load_pipeline_config(PipelineStep.model_definition)

        ### get config values ###
        d_model: int = int(
            get_config_value(config, "transformer_embedding", "d_model", 192)
        )
        if vocab_size is None:
            from src.ms2.models.rnn.embedder import get_vocab_size

            vocab_size = get_vocab_size()
        resolved_vocab_size: int = vocab_size

        ### store dimension needed at call time ###
        self.d_model = d_model

        ### define the token embedding table ###
        self.embedding = tf.keras.layers.Embedding(
            input_dim=resolved_vocab_size,
            output_dim=d_model,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                stddev=d_model**-0.5
            ),
            mask_zero=True,
            name="transformer_token_embedding",
        )

    def call(self, token_ids: tf.Tensor) -> tf.Tensor:
        """
        Embed token IDs and apply ``sqrt(d_model)`` scaling.

        Args:
            token_ids: Integer token ID tensor of shape ``(B, L)``.

        Returns:
            Embedded tensor of shape ``(B, L, d_model)``.
        """
        ### get the raw token embeddings ###
        # dim(x) = (B, L, d_model)
        x = self.embedding(token_ids)

        ### apply standard transformer scaling ###
        # dim(output) = (B, L, d_model)
        output = x * tf.cast(math.sqrt(self.d_model), dtype=x.dtype)

        return output
