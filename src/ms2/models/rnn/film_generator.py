from __future__ import annotations

import tensorflow as tf

D_DEC = 256
D_FILM_H = 128


class _FilmBiasInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape: tuple[int, ...], dtype: tf.dtypes.DType | None = None) -> tf.Tensor:
        if shape[-1] != 2 * D_DEC:
            raise ValueError("FiLM bias last dimension must be 2*d_dec")
        return tf.concat([tf.ones((D_DEC,), dtype=dtype), tf.zeros((D_DEC,), dtype=dtype)], axis=0)


class FiLMGenerator(tf.keras.layers.Layer):
    """Static global FiLM generator with identity init, ADR §2.10."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.hidden = tf.keras.layers.Dense(D_FILM_H, activation=tf.keras.activations.gelu, name="film_hidden")
        self.out = tf.keras.layers.Dense(
            2 * D_DEC,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer=_FilmBiasInitializer(),
            name="film_gamma_beta",
        )

    def call(self, encoder_summary: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        gamma_beta = self.out(self.hidden(encoder_summary))
        return tf.split(gamma_beta, 2, axis=-1)
