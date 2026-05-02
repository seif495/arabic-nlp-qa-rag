from __future__ import annotations

import tensorflow as tf

from src.ms2.models.transformer.attention import (
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
)
from src.ms2.models.transformer.embedding import (
    D_MODEL,
    TransformerTokenEmbedding,
    sinusoidal_encoding,
)
from src.ms2.models.transformer.encoder import D_FF
from src.ms2.models.transformer.kv_cache import KVCache


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, use_rope: bool = True, dropout_resid: float = 0.1, **kwargs: object
    ) -> None:
        super().__init__(**kwargs)
        self.ln_self = tf.keras.layers.LayerNormalization(name="dec_ln_self")
        self.self_attn = MultiHeadSelfAttention(
            use_rope=use_rope, max_length=64, name="dec_self_attn"
        )
        self.drop_self = tf.keras.layers.Dropout(dropout_resid)
        self.ln_cross = tf.keras.layers.LayerNormalization(name="dec_ln_cross")
        self.cross_attn = MultiHeadCrossAttention(name="dec_cross_attn")
        self.drop_cross = tf.keras.layers.Dropout(dropout_resid)
        self.ln_ffn = tf.keras.layers.LayerNormalization(name="dec_ln_ffn")
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(D_FF, activation=tf.keras.activations.gelu),
                tf.keras.layers.Dense(D_MODEL),
            ],
            name="dec_ffn",
        )
        self.drop_ffn = tf.keras.layers.Dropout(dropout_resid)

    def call(
        self,
        x: tf.Tensor,
        encoder_output: tf.Tensor,
        decoder_padding_mask: tf.Tensor | None,
        encoder_padding_mask: tf.Tensor,
        training: bool = False,
        self_cache: dict[str, tf.Tensor] | None = None,
        cross_cache: dict[str, tf.Tensor] | None = None,
        start: int = 0,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        self_out, self_weights = self.self_attn(
            self.ln_self(x),
            padding_mask=decoder_padding_mask,
            use_causal_mask=self_cache is None,
            training=training,
            cache=self_cache,
            start=start,
        )
        x = x + self.drop_self(self_out, training=training)
        cross_out, cross_weights = self.cross_attn(
            self.ln_cross(x),
            encoder_output,
            encoder_padding_mask=encoder_padding_mask,
            training=training,
            cache=cross_cache,
        )
        x = x + self.drop_cross(cross_out, training=training)
        x = x + self.drop_ffn(
            self.ffn(self.ln_ffn(x), training=training), training=training
        )
        return x, self_weights, cross_weights


class TransformerDecoder(tf.keras.layers.Layer):
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
        layer = DecoderLayer(use_rope=use_rope)
        self.layers_ = (
            [layer] * n_layers
            if shared_layers
            else [
                DecoderLayer(use_rope=use_rope, name=f"decoder_layer_{idx}")
                for idx in range(n_layers)
            ]
        )
        self.final_ln = tf.keras.layers.LayerNormalization(name="decoder_final_ln")

    def call(
        self,
        decoder_ids: tf.Tensor,
        encoder_output: tf.Tensor,
        encoder_padding_mask: tf.Tensor,
        training: bool = False,
        cache: KVCache | None = None,
    ) -> tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor]]:
        decoder_padding_mask = tf.not_equal(decoder_ids, 0)
        x = self.embedding(decoder_ids, training=training)
        if self.positional_mode == "sinusoidal_pe":
            x = (
                x
                + tf.cast(sinusoidal_encoding(tf.shape(decoder_ids)[1]), x.dtype)[
                    None, :, :
                ]
            )
        self_weights_list = []
        cross_weights_list = []
        start = cache.populated_steps() if cache is not None else 0
        for idx, layer in enumerate(self.layers_):
            x, self_weights, cross_weights = layer(
                x,
                encoder_output,
                decoder_padding_mask=None
                if cache is not None
                else decoder_padding_mask,
                encoder_padding_mask=encoder_padding_mask,
                training=training,
                self_cache=None if cache is None else cache.self_attention[idx],
                cross_cache=None if cache is None else cache.cross_attention[idx],
                start=start,
            )
            if (
                cache is not None
                and tf.executing_eagerly()
                and layer.self_attn.last_k is not None
            ):
                cache.self_attention[idx]["k"] = layer.self_attn.last_k
                cache.self_attention[idx]["v"] = layer.self_attn.last_v
            self_weights_list.append(self_weights)
            cross_weights_list.append(cross_weights)
        return self.final_ln(x), self_weights_list, cross_weights_list
