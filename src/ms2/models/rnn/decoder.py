from __future__ import annotations

import tensorflow as tf

from src.ms2.models.rnn.embedding import D_TOK, VOCAB_SIZE, SharedTokenEmbedding
from src.ms2.models.rnn.film_generator import D_DEC


class TiedOutputProjection(tf.keras.layers.Layer):
    """Projection using a live reference to the shared embedding variable."""

    def __init__(
        self,
        embedding: SharedTokenEmbedding,
        vocab_size: int = VOCAB_SIZE,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding = embedding
        self.to_tok = tf.keras.layers.Dense(
            D_TOK, use_bias=False, name="decoder_to_token_dim"
        )
        self.bias = self.add_weight(
            name="output_bias", shape=(vocab_size,), initializer="zeros"
        )

    def call(self, decoder_output: tf.Tensor) -> tf.Tensor:
        projected = self.to_tok(decoder_output)
        logits = tf.matmul(projected, self.embedding.embeddings, transpose_b=True)
        return logits + tf.cast(self.bias, logits.dtype)


class FiLMLSTMDecoder(tf.keras.layers.Layer):
    """Two-layer FiLM-conditioned LSTM decoder from ADR §2.11."""

    def __init__(
        self,
        embedding: SharedTokenEmbedding,
        dropout_emb: float = 0.2,
        dropout_rnn: float = 0.3,
        use_film: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding = embedding
        self.use_film = use_film
        self.dropout_emb = tf.keras.layers.Dropout(dropout_emb)
        self.lstm1 = tf.keras.layers.LSTM(
            D_DEC, return_sequences=True, return_state=True, name="decoder_lstm_1"
        )
        self.lstm2 = tf.keras.layers.LSTM(
            D_DEC, return_sequences=True, return_state=True, name="decoder_lstm_2"
        )
        self.dropout_rnn = tf.keras.layers.Dropout(dropout_rnn)
        self.h_init = [
            tf.keras.layers.Dense(
                D_DEC, activation="tanh", name=f"decoder_h_init_{idx}"
            )
            for idx in range(2)
        ]
        self.c_init = [
            tf.keras.layers.Dense(
                D_DEC, activation="tanh", name=f"decoder_c_init_{idx}"
            )
            for idx in range(2)
        ]
        self.output_projection = TiedOutputProjection(
            embedding, name="decoder_tied_output"
        )

    def initial_states(self, encoder_summary: tf.Tensor) -> list[list[tf.Tensor]]:
        return [
            [self.h_init[idx](encoder_summary), self.c_init[idx](encoder_summary)]
            for idx in range(2)
        ]

    def call(
        self,
        decoder_inputs: tf.Tensor,
        encoder_summary: tf.Tensor,
        gamma: tf.Tensor,
        beta: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        states = self.initial_states(encoder_summary)
        x = self.dropout_emb(self.embedding(decoder_inputs), training=training)
        x, *_ = self.lstm1(x, initial_state=states[0], training=training)
        x = self._film(x, gamma, beta)
        x = self.dropout_rnn(x, training=training)
        x, *_ = self.lstm2(x, initial_state=states[1], training=training)
        x = self._film(x, gamma, beta)
        return self.output_projection(x)

    def _film(self, x: tf.Tensor, gamma: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
        if not self.use_film:
            return x
        return x * gamma[:, None, :] + beta[:, None, :]
