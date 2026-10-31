"""NumPy inference port of MS2 Model A (``src/ms2/models/rnn/model_a.py``, ADR §2).

Tri-encoder recurrent QA model with no attention:

- Branch 1: question Bi-LSTM + learned (masked-softmax) pooling;
- Branch 2: two-layer context Bi-GRU;
- Branch 3: joint ``[BOS] q [SEP] c [EOS]`` tokens + per-token char-CNN, Bi-LSTM;
- branches aligned to the context axis, fused by a per-position 3-expert
  softmax gate, refined by a Bi-GRU whose final states form the encoder summary;
- decoder: two-layer LSTM initialised from the summary, each layer's output
  FiLM-modulated (``gamma``, ``beta`` from the summary), tied output projection.

Keras semantics reproduced: LSTM gates ``i, f, c, o``; GRU ``reset_after=True``
gates ``z, r, h``; masked timesteps keep the previous state and (inside
``Bidirectional``) emit zeros. ``tests/web/test_rnn_parity.py`` checks this
port against TensorFlow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ms3web.seq2seq import BOS_ID, EOS_ID, NEG_INF, DecodeResult, Dense, LayerNorm, gelu


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Recurrent cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LSTMCell:
    kernel: np.ndarray
    recurrent: np.ndarray
    bias: np.ndarray

    @property
    def units(self) -> int:
        return self.recurrent.shape[0]

    def step(self, x_proj: np.ndarray, h: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = x_proj + h @ self.recurrent
        i, f, g, o = np.split(z, 4, axis=-1)
        c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
        return sigmoid(o) * np.tanh(c), c


@dataclass(frozen=True)
class GRUCell:
    kernel: np.ndarray
    recurrent: np.ndarray
    bias: np.ndarray  # (2, 3u): input bias, recurrent bias

    @property
    def units(self) -> int:
        return self.recurrent.shape[0]

    def step(self, x_proj: np.ndarray, h: np.ndarray) -> np.ndarray:
        h_proj = h @ self.recurrent + self.bias[1]
        xz, xr, xh = np.split(x_proj, 3, axis=-1)
        hz, hr, hh = np.split(h_proj, 3, axis=-1)
        z = sigmoid(xz + hz)
        r = sigmoid(xr + hr)
        candidate = np.tanh(xh + r * hh)
        return z * h + (1.0 - z) * candidate


def run_rnn(
    cell: LSTMCell | GRUCell,
    x: np.ndarray,
    mask: np.ndarray | None,
    reverse: bool = False,
    initial: tuple[np.ndarray, ...] | None = None,
    zero_masked_output: bool = True,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Run ``cell`` over ``x[B, T, F]``; masked steps keep state (Keras semantics)."""
    batch, steps, _ = x.shape
    units = cell.units
    is_lstm = isinstance(cell, LSTMCell)
    bias = cell.bias if is_lstm else cell.bias[0]
    x_proj = x @ cell.kernel + bias
    if initial is not None:
        state = tuple(initial)
    else:
        state = (np.zeros((batch, units), np.float32),) * (2 if is_lstm else 1)
    out = np.zeros((batch, steps, units), np.float32)
    prev_out = np.zeros((batch, units), np.float32)
    order = range(steps - 1, -1, -1) if reverse else range(steps)
    for t in order:
        if is_lstm:
            new_state = cell.step(x_proj[:, t], state[0], state[1])
        else:
            new_state = (cell.step(x_proj[:, t], state[0]),)
        if mask is None:
            state = new_state
            out[:, t] = state[0]
            prev_out = state[0]
            continue
        keep = mask[:, t][:, None]
        state = tuple(np.where(keep, new, old) for new, old in zip(new_state, state, strict=True))
        if zero_masked_output:
            out[:, t] = np.where(keep, state[0], 0.0)
        else:
            out[:, t] = np.where(keep, state[0], prev_out)
            prev_out = out[:, t]
    return out, state


@dataclass(frozen=True)
class BiRNN:
    forward: LSTMCell | GRUCell
    backward: LSTMCell | GRUCell

    def __call__(self, x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fwd, fwd_state = run_rnn(self.forward, x, mask)
        bwd, bwd_state = run_rnn(self.backward, x, mask, reverse=True)
        return np.concatenate([fwd, bwd], axis=-1), fwd_state[0], bwd_state[0]


# ---------------------------------------------------------------------------
# Model A
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderOutput:
    summary: np.ndarray  # (B, 192)
    question_pooled: np.ndarray  # Branch 1 pooled question (B, 192)
    context_seq: np.ndarray  # Branch 2 context states (B, L_c, 192)
    context_mask: np.ndarray  # (B, L_c)


class ModelANumpy:
    """Inference-only Model A. Build with :meth:`load` from an exported ``.npz``."""

    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        meta = {k[len("meta.") :]: float(v) for k, v in w.items() if k.startswith("meta.")}
        self.eps = meta["ln_epsilon"]
        self.use_film = bool(meta["use_film"])
        self.dec_max_length = int(meta["decoder_max_length"])
        self.embedding = w["embedding"]
        self.char_embedding = w["char_embedding"]
        self.char_convs = [
            (w[f"char_conv.{k}.kernel"], w[f"char_conv.{k}.bias"])
            for k in range(int(meta["n_char_convs"]))
        ]

        def lstm(name: str) -> LSTMCell:
            return LSTMCell(w[f"{name}.kernel"], w[f"{name}.recurrent_kernel"], w[f"{name}.bias"])

        def gru(name: str) -> GRUCell:
            return GRUCell(w[f"{name}.kernel"], w[f"{name}.recurrent_kernel"], w[f"{name}.bias"])

        def dense(name: str) -> Dense:
            bias = w.get(f"{name}.bias")
            kernel = w[f"{name}.kernel"]
            return Dense(kernel, bias if bias is not None else np.zeros(kernel.shape[1], np.float32))

        self.b1 = BiRNN(lstm("branch1.forward"), lstm("branch1.backward"))
        self.pool_proj = dense("branch1.pool_proj")
        self.pool_score = dense("branch1.pool_score")
        self.b2_1 = BiRNN(gru("branch2.gru1.forward"), gru("branch2.gru1.backward"))
        self.b2_2 = BiRNN(gru("branch2.gru2.forward"), gru("branch2.gru2.backward"))
        self.b3 = BiRNN(lstm("branch3.forward"), lstm("branch3.backward"))
        self.merge_proj = [dense(f"merge.proj{i}") for i in (1, 2, 3)]
        self.gate_hidden = dense("merge.gate_hidden")
        self.gate_out = dense("merge.gate_out")
        self.merge_norm = LayerNorm(w["merge.norm.gamma"], w["merge.norm.beta"], self.eps)
        self.refine = BiRNN(gru("refine.forward"), gru("refine.backward"))
        self.film_hidden = dense("film.hidden")
        self.film_out = dense("film.out")
        self.h_init = [dense(f"decoder.h_init_{i}") for i in range(2)]
        self.c_init = [dense(f"decoder.c_init_{i}") for i in range(2)]
        self.dec_lstm = [lstm("decoder.lstm1"), lstm("decoder.lstm2")]
        self.to_tok = dense("decoder.to_tok")
        self.output_bias = w["decoder.output_bias"]

    @classmethod
    def load(cls, path: str | Path) -> ModelANumpy:
        with np.load(path) as data:
            return cls({key: data[key] for key in data.files})

    # -- encoder ------------------------------------------------------------

    def question_branch(self, question_ids: np.ndarray) -> np.ndarray:
        mask = question_ids != 0
        seq, _, _ = self.b1(self.embedding[question_ids], mask)
        scores = self.pool_score(np.tanh(self.pool_proj(seq)))[..., 0]
        weights = softmax(np.where(mask, scores, NEG_INF), axis=-1)
        return (seq * weights[..., None]).sum(axis=1)

    def context_branch(self, context_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = context_ids != 0
        x, _, _ = self.b2_1(self.embedding[context_ids], mask)
        x, _, _ = self.b2_2(x, mask)
        return x, mask

    def char_cnn(self, char_ids: np.ndarray) -> np.ndarray:
        batch, steps, chars = char_ids.shape
        emb = self.char_embedding[char_ids.reshape(batch * steps, chars)]  # (N, C, d_char)
        pooled = []
        for kernel, bias in self.char_convs:
            width = kernel.shape[0]
            windows = np.stack([emb[:, i : i + width] for i in range(chars - width + 1)], axis=1)
            conv = np.einsum("nwkc,kcf->nwf", windows, kernel) + bias
            pooled.append(np.maximum(conv, 0.0).max(axis=1))
        return np.concatenate(pooled, axis=-1).reshape(batch, steps, -1)

    def joint_branch(self, joint_ids: np.ndarray, char_ids: np.ndarray) -> np.ndarray:
        mask = joint_ids != 0
        x = np.concatenate([self.embedding[joint_ids], self.char_cnn(char_ids)], axis=-1)
        seq, _, _ = self.b3(x, mask)
        return seq

    def encode(
        self,
        question_ids: np.ndarray,
        context_ids: np.ndarray,
        joint_ids: np.ndarray,
        char_ids: np.ndarray,
    ) -> EncoderOutput:
        b1 = self.question_branch(question_ids)
        b2, context_mask = self.context_branch(context_ids)
        b3 = self.joint_branch(joint_ids, char_ids)
        l_c = b2.shape[1]
        # Align Branch 3 to the context axis: context starts after [BOS] q [SEP].
        starts = (question_ids != 0).sum(axis=1) + 2
        offsets = starts[:, None] + np.arange(l_c)[None, :]
        b3_aligned = np.take_along_axis(b3, offsets[..., None], axis=1)
        h1 = self.merge_proj[0](np.repeat(b1[:, None, :], l_c, axis=1))
        h2 = self.merge_proj[1](b2)
        h3 = self.merge_proj[2](b3_aligned)
        gates = softmax(self.gate_out(gelu(self.gate_hidden(np.concatenate([h1, h2, h3], axis=-1)))))
        fused = self.merge_norm(gates[..., 0:1] * h1 + gates[..., 1:2] * h2 + gates[..., 2:3] * h3)
        _, fwd_h, bwd_h = self.refine(fused, context_mask)
        return EncoderOutput(np.concatenate([fwd_h, bwd_h], axis=-1), b1, b2, context_mask)

    # -- decoder ------------------------------------------------------------

    def _film(self, summary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gamma_beta = self.film_out(gelu(self.film_hidden(summary)))
        return np.split(gamma_beta, 2, axis=-1)

    def _project(self, x: np.ndarray) -> np.ndarray:
        return self.to_tok(x) @ self.embedding.T + self.output_bias

    def decoder_logits(self, summary: np.ndarray, decoder_ids: np.ndarray) -> np.ndarray:
        """Teacher-forced logits ``[B, L_dec, vocab]``."""
        gamma, beta = self._film(summary)
        x = self.embedding[decoder_ids]
        for idx, cell in enumerate(self.dec_lstm):
            init = (np.tanh(self.h_init[idx](summary)), np.tanh(self.c_init[idx](summary)))
            x, _ = run_rnn(cell, x, decoder_ids != 0, initial=init, zero_masked_output=False)
            if self.use_film:
                x = x * gamma[:, None, :] + beta[:, None, :]
        return self._project(x)

    def greedy_decode(
        self, summary: np.ndarray, max_length: int | None = None, max_repeat: int | None = 3
    ) -> list[DecodeResult]:
        """Batched greedy decoding from ``summary[B, 192]``."""
        max_length = min(max_length or self.dec_max_length, self.dec_max_length)
        batch = summary.shape[0]
        gamma, beta = self._film(summary)
        states = [
            (np.tanh(self.h_init[i](summary)), np.tanh(self.c_init[i](summary))) for i in range(2)
        ]
        tokens = np.full(batch, BOS_ID)
        outputs: list[list[int]] = [[] for _ in range(batch)]
        log_probs: list[list[float]] = [[] for _ in range(batch)]
        done = np.zeros(batch, dtype=bool)
        for _ in range(max_length):
            x = self.embedding[tokens]
            for idx, cell in enumerate(self.dec_lstm):
                h, c = cell.step(x @ cell.kernel + cell.bias, *states[idx])
                states[idx] = (h, c)
                x = h * gamma + beta if self.use_film else h
            logits = self._project(x)
            shifted = logits - logits.max(axis=-1, keepdims=True)
            log_softmax = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
            tokens = logits.argmax(axis=-1)
            for b in np.flatnonzero(~done):
                token = int(tokens[b])
                log_probs[b].append(float(log_softmax[b, token]))
                repeated = max_repeat and outputs[b][-max_repeat:] == [token] * max_repeat
                if token == EOS_ID or repeated:
                    done[b] = True
                else:
                    outputs[b].append(token)
            if done.all():
                break
        return [
            DecodeResult(out, float(np.mean(lp)) if lp else float("-inf"))
            for out, lp in zip(outputs, log_probs, strict=True)
        ]
