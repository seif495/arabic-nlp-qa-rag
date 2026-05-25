from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tensorflow as tf

from src.ms2.inference.greedy import DecodableModel


@dataclass
class _ModelAState:
    encoder_summary: tf.Tensor
    gamma: tf.Tensor
    beta: tf.Tensor
    lstm1_h: tf.Tensor
    lstm1_c: tf.Tensor
    lstm2_h: tf.Tensor
    lstm2_c: tf.Tensor


class ModelADecodableAdapter:
    """Wraps ModelA to satisfy the DecodableModel protocol for step-by-step decoding."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def init_state(self, encoder_inputs: dict[str, tf.Tensor]) -> _ModelAState:
        m = self.model
        question_ids = encoder_inputs["question_ids"]
        context_ids = encoder_inputs["context_ids"]
        joint_ids = encoder_inputs["encoder_input_ids"]
        char_ids = encoder_inputs["char_matrix"]

        question_lengths = tf.reduce_sum(
            tf.cast(tf.not_equal(question_ids, 0), tf.int32), axis=1
        )
        b1_pooled, _ = m.branch1(question_ids, training=False)
        b2_seq, context_mask = m.branch2(context_ids, training=False)
        b3_seq, _ = m.branch3(joint_ids, char_ids, training=False)
        aligned = m.aligner(b1_pooled, b2_seq, b3_seq, question_lengths)
        fused, _ = m.merge(aligned, training=False)
        _, encoder_summary = m.refinement(fused, mask=context_mask, training=False)
        gamma, beta = m.film(encoder_summary)

        initial_states = m.decoder.initial_states(encoder_summary)
        return _ModelAState(
            encoder_summary=encoder_summary,
            gamma=gamma,
            beta=beta,
            lstm1_h=initial_states[0][0],
            lstm1_c=initial_states[0][1],
            lstm2_h=initial_states[1][0],
            lstm2_c=initial_states[1][1],
        )

    def step(
        self, state: _ModelAState, last_token: int
    ) -> tuple[list[float], _ModelAState]:
        dec = self.model.decoder
        x = dec.embedding(tf.constant([[last_token]], dtype=tf.int32))  # (1, 1, D_TOK)
        out1, h1, c1 = dec.lstm1(
            x, initial_state=[state.lstm1_h, state.lstm1_c], training=False
        )
        out1 = out1 * state.gamma[:, None, :] + state.beta[:, None, :]
        out2, h2, c2 = dec.lstm2(
            out1, initial_state=[state.lstm2_h, state.lstm2_c], training=False
        )
        out2 = out2 * state.gamma[:, None, :] + state.beta[:, None, :]
        logits = dec.output_projection(out2)  # (1, 1, VOCAB_SIZE)
        new_state = _ModelAState(
            encoder_summary=state.encoder_summary,
            gamma=state.gamma,
            beta=state.beta,
            lstm1_h=h1,
            lstm1_c=c1,
            lstm2_h=h2,
            lstm2_c=c2,
        )
        return logits[0, 0, :].numpy().tolist(), new_state


@dataclass
class _ModelBState:
    encoder_output: tf.Tensor
    enc_mask: tf.Tensor
    tokens: list[int] = field(default_factory=list)


class ModelBDecodableAdapter:
    """Wraps ModelB to satisfy the DecodableModel protocol for step-by-step decoding."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def init_state(self, encoder_inputs: dict[str, tf.Tensor]) -> _ModelBState:
        encoder_ids = encoder_inputs["encoder_input_ids"]
        encoder_output, _, enc_mask = self.model.encoder(encoder_ids, training=False)
        return _ModelBState(encoder_output=encoder_output, enc_mask=enc_mask)

    def step(
        self, state: _ModelBState, last_token: int
    ) -> tuple[list[float], _ModelBState]:
        tokens = state.tokens + [last_token]
        decoder_ids = tf.constant([tokens], dtype=tf.int32)  # (1, t)
        decoder_output, _, _ = self.model.decoder(
            decoder_ids, state.encoder_output, state.enc_mask, training=False
        )
        logits = (
            tf.matmul(
                decoder_output, self.model.embedding.embeddings, transpose_b=True
            )
            + tf.cast(self.model.output_bias, decoder_output.dtype)
        )  # (1, t, VOCAB_SIZE)
        new_state = _ModelBState(
            encoder_output=state.encoder_output,
            enc_mask=state.enc_mask,
            tokens=tokens,
        )
        return logits[0, -1, :].numpy().tolist(), new_state


def make_decodable_adapter(model: Any, model_token: str) -> DecodableModel:
    if model_token == "a":
        return ModelADecodableAdapter(model)
    return ModelBDecodableAdapter(model)
