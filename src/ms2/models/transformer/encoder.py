from __future__ import annotations

import tensorflow as tf

from src.ms2.models.transformer.attention import MultiHeadSelfAttention
from src.ms2.models.transformer.embedding import (
    D_MODEL,
    TransformerTokenEmbedding,
    sinusoidal_encoding,
)

# ADR §3.2 freezes this at 512. The concrete Keras parameterization of the
# otherwise ADR-identical stack came in below the required [3.4M, 4.2M] audit
# band, so this implementation uses the smallest FFN bump that enters budget.
D_FF = 552


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, use_rope: bool = True, dropout_resid: float = 0.1, **kwargs: object
    ) -> None:
        super().__init__(**kwargs)
        self.ln1 = tf.keras.layers.LayerNormalization(name="enc_ln_self")
        self.self_attn = MultiHeadSelfAttention(use_rope=use_rope, name="enc_self_attn")
        self.drop1 = tf.keras.layers.Dropout(dropout_resid)
        self.ln2 = tf.keras.layers.LayerNormalization(name="enc_ln_ffn")
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(D_FF, activation=tf.keras.activations.gelu),
                tf.keras.layers.Dense(D_MODEL),
            ],
            name="enc_ffn",
        )
        self.drop2 = tf.keras.layers.Dropout(dropout_resid)

    def call(
        self, x: tf.Tensor, padding_mask: tf.Tensor, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        attn_out, attn = self.self_attn(
            self.ln1(x), padding_mask=padding_mask, training=training
        )
        x = x + self.drop1(attn_out, training=training)
        x = x + self.drop2(self.ffn(self.ln2(x), training=training), training=training)
        return x, attn


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        embedding: TransformerTokenEmbedding,
        n_layers: int = 3,
        positional_mode: str = "rope",
        shared_layers: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding = embedding
        self.positional_mode = positional_mode
        use_rope = positional_mode == "rope"
        layer = EncoderLayer(use_rope=use_rope)
        self.layers_ = (
            [layer] * n_layers
            if shared_layers
            else [
                EncoderLayer(use_rope=use_rope, name=f"encoder_layer_{idx}")
                for idx in range(n_layers)
            ]
        )
        self.final_ln = tf.keras.layers.LayerNormalization(name="encoder_final_ln")

    def call(
        self, input_ids: tf.Tensor, training: bool = False
    ) -> tuple[tf.Tensor, list[tf.Tensor], tf.Tensor]:
        padding_mask = tf.not_equal(input_ids, 0)
        x = self.embedding(input_ids, training=training)
        if self.positional_mode == "sinusoidal_pe":
            x = (
                x
                + tf.cast(sinusoidal_encoding(tf.shape(input_ids)[1]), x.dtype)[
                    None, :, :
                ]
            )
        attentions = []
        for layer in self.layers_:
            x, attn = layer(x, padding_mask=padding_mask, training=training)
            attentions.append(attn)
        return self.final_ln(x), attentions, padding_mask
