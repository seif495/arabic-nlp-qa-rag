from __future__ import annotations

import tensorflow as tf

D_FUSE = 192


class GatedMerge(tf.keras.layers.Layer):
    """Per-position three-expert softmax MoE from ADR §2.8."""

    def __init__(
        self, dropout_ff: float = 0.3, mean_merge: bool = False, **kwargs: object
    ) -> None:
        super().__init__(**kwargs)
        self.mean_merge = mean_merge
        self.proj1 = tf.keras.layers.Dense(D_FUSE, name="merge_proj_branch1")
        self.proj2 = tf.keras.layers.Dense(D_FUSE, name="merge_proj_branch2")
        self.proj3 = tf.keras.layers.Dense(D_FUSE, name="merge_proj_branch3")
        self.gate_hidden = tf.keras.layers.Dense(
            D_FUSE, activation=tf.keras.activations.gelu, name="merge_gate_hidden"
        )
        self.gate_out = tf.keras.layers.Dense(3, name="merge_gate_logits")
        self.dropout = tf.keras.layers.Dropout(dropout_ff)
        self.norm = tf.keras.layers.LayerNormalization(name="merge_layer_norm")

    def call(
        self, branches: tuple[tf.Tensor, tf.Tensor, tf.Tensor], training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        h1 = self.proj1(branches[0])
        h2 = self.proj2(branches[1])
        h3 = self.proj3(branches[2])
        if self.mean_merge:
            gate_shape = tf.concat([tf.shape(h1)[:-1], tf.constant([3])], axis=0)
            gates = tf.ones(gate_shape, dtype=h1.dtype) / tf.cast(3.0, h1.dtype)
        else:
            hidden = self.gate_hidden(tf.concat([h1, h2, h3], axis=-1))
            gates = tf.nn.softmax(tf.cast(self.gate_out(hidden), tf.float32), axis=-1)
            gates = tf.cast(gates, h1.dtype)
        fused = gates[..., 0:1] * h1 + gates[..., 1:2] * h2 + gates[..., 2:3] * h3
        return self.norm(self.dropout(fused, training=training)), gates
