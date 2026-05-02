from __future__ import annotations

import math

import tensorflow as tf

from src.ms2.models.transformer.embedding import D_MODEL
from src.ms2.models.transformer.rope import D_HEAD, RoPE

N_HEADS = 4


def causal_mask(length: tf.Tensor | int) -> tf.Tensor:
    return tf.linalg.band_part(tf.ones((length, length), dtype=tf.bool), -1, 0)


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """Direct matmul self-attention with RoPE on Q/K only."""

    def __init__(
        self,
        use_rope: bool = True,
        dropout_attn: float = 0.1,
        max_length: int = 420,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.use_rope = use_rope
        self.wq = tf.keras.layers.Dense(D_MODEL, name="self_wq")
        self.wk = tf.keras.layers.Dense(D_MODEL, name="self_wk")
        self.wv = tf.keras.layers.Dense(D_MODEL, name="self_wv")
        self.wo = tf.keras.layers.Dense(D_MODEL, name="self_wo")
        self.rope = RoPE(max_length=max_length) if use_rope else None
        self.dropout = tf.keras.layers.Dropout(dropout_attn)
        self.last_attention: tf.Tensor | None = None
        self.softmax_dtype: tf.dtypes.DType | None = None
        self.last_k: tf.Tensor | None = None
        self.last_v: tf.Tensor | None = None

    def call(
        self,
        x: tf.Tensor,
        padding_mask: tf.Tensor | None = None,
        use_causal_mask: bool = False,
        training: bool = False,
        cache: dict[str, tf.Tensor] | None = None,
        start: int = 0,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        q = _split_heads(self.wq(x))
        k = _split_heads(self.wk(x))
        v = _split_heads(self.wv(x))
        if self.rope is not None:
            q = self.rope(q, start=start)
            k = self.rope(k, start=start)
        if cache is not None and tf.executing_eagerly():
            if "k" in cache:
                k = tf.concat([cache["k"], k], axis=2)
                v = tf.concat([cache["v"], v], axis=2)
            cache["k"], cache["v"] = k, v
        self.last_k = k
        self.last_v = v
        scores = tf.matmul(q, k, transpose_b=True) / math.sqrt(D_HEAD)
        scores = _apply_masks(
            scores, padding_mask=padding_mask, use_causal_mask=use_causal_mask
        )
        attn = tf.nn.softmax(tf.cast(scores, tf.float32), axis=-1)
        self.softmax_dtype = attn.dtype
        attn = tf.cast(attn, v.dtype)
        self.last_attention = attn
        out = tf.matmul(self.dropout(attn, training=training), v)
        return self.wo(_merge_heads(out)), attn


class MultiHeadCrossAttention(tf.keras.layers.Layer):
    """Direct matmul cross-attention with no RoPE, ADR §3.4.5."""

    def __init__(self, dropout_attn: float = 0.1, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.wq = tf.keras.layers.Dense(D_MODEL, name="cross_wq")
        self.wk = tf.keras.layers.Dense(D_MODEL, name="cross_wk")
        self.wv = tf.keras.layers.Dense(D_MODEL, name="cross_wv")
        self.wo = tf.keras.layers.Dense(D_MODEL, name="cross_wo")
        self.dropout = tf.keras.layers.Dropout(dropout_attn)
        self.rope = None
        self.last_attention: tf.Tensor | None = None

    def call(
        self,
        x: tf.Tensor,
        encoder_output: tf.Tensor,
        encoder_padding_mask: tf.Tensor | None = None,
        training: bool = False,
        cache: dict[str, tf.Tensor] | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        q = _split_heads(self.wq(x))
        if cache is not None and "k" in cache:
            k, v = cache["k"], cache["v"]
        else:
            k = _split_heads(self.wk(encoder_output))
            v = _split_heads(self.wv(encoder_output))
            if cache is not None:
                cache["k"], cache["v"] = k, v
        scores = tf.matmul(q, k, transpose_b=True) / math.sqrt(D_HEAD)
        scores = _apply_masks(
            scores, padding_mask=encoder_padding_mask, use_causal_mask=False
        )
        attn = tf.cast(tf.nn.softmax(tf.cast(scores, tf.float32), axis=-1), v.dtype)
        self.last_attention = attn
        return self.wo(
            _merge_heads(tf.matmul(self.dropout(attn, training=training), v))
        ), attn


def _split_heads(x: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(x)
    x = tf.reshape(x, (shape[0], shape[1], N_HEADS, D_HEAD))
    return tf.transpose(x, (0, 2, 1, 3))


def _merge_heads(x: tf.Tensor) -> tf.Tensor:
    x = tf.transpose(x, (0, 2, 1, 3))
    shape = tf.shape(x)
    return tf.reshape(x, (shape[0], shape[1], D_MODEL))


def _apply_masks(
    scores: tf.Tensor, padding_mask: tf.Tensor | None, use_causal_mask: bool
) -> tf.Tensor:
    if padding_mask is not None:
        key_mask = tf.cast(padding_mask[:, None, None, :], tf.bool)
        scores = tf.where(
            key_mask, scores, tf.fill(tf.shape(scores), tf.constant(-1e9, scores.dtype))
        )
    if use_causal_mask:
        mask = causal_mask(tf.shape(scores)[-2])
        scores = tf.where(
            mask[None, None, :, :],
            scores,
            tf.fill(tf.shape(scores), tf.constant(-1e9, scores.dtype)),
        )
    return scores
