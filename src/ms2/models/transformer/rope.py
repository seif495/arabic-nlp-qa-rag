from __future__ import annotations

import tensorflow as tf

D_HEAD = 48
THETA_BASE = 10000.0


def rope_frequencies(d_head: int = D_HEAD, theta_base: float = THETA_BASE) -> tf.Tensor:
    idx = tf.range(d_head // 2, dtype=tf.float32)
    return tf.pow(tf.constant(theta_base, tf.float32), -2.0 * idx / float(d_head))


def apply_rope(x: tf.Tensor, cos: tf.Tensor, sin: tf.Tensor) -> tf.Tensor:
    """Apply RoPE to arbitrary leading dims ending in `(L, d_head)`."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    return tf.reshape(tf.stack([rot_even, rot_odd], axis=-1), tf.shape(x))


class RoPE(tf.keras.layers.Layer):
    """Rotary positional embeddings precomputed at build/init time, ADR §3.4."""

    def __init__(self, max_length: int = 420, d_head: int = D_HEAD, theta_base: float = THETA_BASE, **kwargs: object) -> None:
        super().__init__(trainable=False, **kwargs)
        self.max_length = max_length
        self.d_head = d_head
        freqs = rope_frequencies(d_head=d_head, theta_base=theta_base)
        positions = tf.range(max_length, dtype=tf.float32)[:, None]
        angles = positions * freqs[None, :]
        self.cos_table = tf.Variable(tf.cos(angles), trainable=False, name="rope_cos")
        self.sin_table = tf.Variable(tf.sin(angles), trainable=False, name="rope_sin")

    def tables(self, length: tf.Tensor | int, start: int = 0) -> tuple[tf.Tensor, tf.Tensor]:
        cos = self.cos_table[start : start + length]
        sin = self.sin_table[start : start + length]
        return cos[None, None, :, :], sin[None, None, :, :]

    def call(self, x: tf.Tensor, start: int = 0) -> tf.Tensor:
        cos, sin = self.tables(tf.shape(x)[-2], start=start)
        return apply_rope(x, tf.cast(cos, x.dtype), tf.cast(sin, x.dtype))
