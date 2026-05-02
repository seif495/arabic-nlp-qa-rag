from __future__ import annotations

import tensorflow as tf

D_REFINE = 96


class RefinementBiGRU(tf.keras.layers.Layer):
    """Single Bi-GRU refinement layer producing output and summary, ADR §2.9."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.bigru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(D_REFINE, return_sequences=True, return_state=True),
            name="refinement_bigru",
        )

    def call(
        self, fused: tf.Tensor, mask: tf.Tensor | None = None, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        output, fwd_state, bwd_state = self.bigru(fused, mask=mask, training=training)
        return output, tf.concat([fwd_state, bwd_state], axis=-1)
