"""NumPy inference port of MS2 Model B (RoPE encoder-decoder Transformer).

TensorFlow does not fit inside a Vercel Python function, so the website runs
Model B with the weights exported by ``python -m src.cli.ms3 export-web``.
The maths mirrors ``src/ms2/models/transformer`` exactly (ADR §3):

- shared token embedding scaled by ``sqrt(d_model)``, tied output projection;
- pre-LN encoder/decoder layers with exact-erf GELU feed-forward blocks;
- RoPE (``d_head = 48``, ``theta = 10000``) rotates Q/K of encoder
  self-attention and decoder masked self-attention; V and cross-attention
  are never rotated (``docs/fs/ms2/ms2-rope-contract.md``).

``tests/web/test_model_b_parity.py`` checks this port against the TensorFlow
model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SEP_ID = 4
NEG_INF = -1e9


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def rope_frequencies(d_head: int, theta_base: float) -> np.ndarray:
    idx = np.arange(d_head // 2, dtype=np.float32)
    return np.power(np.float32(theta_base), -2.0 * idx / float(d_head)).astype(
        np.float32
    )


class RoPE:
    """Precomputed cos/sin tables; mirrors ``src/ms2/models/transformer/rope.py``."""

    def __init__(self, max_length: int, d_head: int, theta_base: float) -> None:
        self.max_length = max_length
        freqs = rope_frequencies(d_head, theta_base)
        angles = np.arange(max_length, dtype=np.float32)[:, None] * freqs[None, :]
        self.cos = np.cos(angles).astype(np.float32)
        self.sin = np.sin(angles).astype(np.float32)

    def __call__(self, x: np.ndarray, start: int = 0) -> np.ndarray:
        """Rotate consecutive (even, odd) pairs of ``x[..., L, d_head]``."""
        length = x.shape[-2]
        if start + length > self.max_length:
            raise ValueError(
                f"RoPE table holds {self.max_length} positions, "
                f"got start={start} length={length}"
            )
        cos = self.cos[start : start + length]
        sin = self.sin[start : start + length]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out


# ---------------------------------------------------------------------------
# Primitive ops
# ---------------------------------------------------------------------------


def _erf(x: np.ndarray) -> np.ndarray:
    # Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7) - avoids a SciPy dependency.
    sign = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    return sign * (1.0 - poly * np.exp(-a * a))


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU, matching ``tf.keras.activations.gelu`` defaults."""
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class Dense:
    w: np.ndarray
    b: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b


@dataclass(frozen=True)
class LayerNorm:
    gamma: np.ndarray
    beta: np.ndarray
    eps: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps) * self.gamma + self.beta


@dataclass(frozen=True)
class FeedForward:
    inner: Dense
    outer: Dense

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.outer(gelu(self.inner(x)))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Attention:
    def __init__(
        self,
        wq: Dense,
        wk: Dense,
        wv: Dense,
        wo: Dense,
        n_heads: int,
        rope: RoPE | None,
    ) -> None:
        self.wq, self.wk, self.wv, self.wo = wq, wk, wv, wo
        self.n_heads = n_heads
        self.rope = rope

    def split(self, x: np.ndarray) -> np.ndarray:
        batch, length, d_model = x.shape
        d_head = d_model // self.n_heads
        return x.reshape(batch, length, self.n_heads, d_head).transpose(0, 2, 1, 3)

    @staticmethod
    def merge(x: np.ndarray) -> np.ndarray:
        batch, heads, length, d_head = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, length, heads * d_head)

    def attend(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None
    ) -> np.ndarray:
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(q.shape[-1])
        if mask is not None:
            scores = np.where(mask, scores, NEG_INF)
        return self.wo(self.merge(softmax(scores) @ v))

    def self_attention(
        self,
        x: np.ndarray,
        key_mask: np.ndarray | None,
        causal: bool,
        start: int = 0,
        cache: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        q, k, v = self.split(self.wq(x)), self.split(self.wk(x)), self.split(self.wv(x))
        if self.rope is not None:
            q = self.rope(q, start=start)
            k = self.rope(k, start=start)
        if cache is not None:
            if "k" in cache:
                k = np.concatenate([cache["k"], k], axis=2)
                v = np.concatenate([cache["v"], v], axis=2)
            cache["k"], cache["v"] = k, v
        mask = None
        if key_mask is not None:
            mask = key_mask[:, None, None, :]
        if causal:
            length = q.shape[2]
            causal_mask = np.tril(np.ones((length, length), dtype=bool))
            mask = causal_mask[None, None] if mask is None else mask & causal_mask
        return self.attend(q, k, v, mask)

    def cross_attention(
        self,
        x: np.ndarray,
        memory: np.ndarray,
        memory_mask: np.ndarray,
        cache: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        q = self.split(self.wq(x))
        if cache is not None and "k" in cache:
            k, v = cache["k"], cache["v"]
        else:
            k, v = self.split(self.wk(memory)), self.split(self.wv(memory))
            if cache is not None:
                cache["k"], cache["v"] = k, v
        return self.attend(q, k, v, memory_mask[:, None, None, :])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderLayer:
    ln_self: LayerNorm
    self_attn: Attention
    ln_ffn: LayerNorm
    ffn: FeedForward

    def __call__(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        x = x + self.self_attn.self_attention(self.ln_self(x), mask, causal=False)
        return x + self.ffn(self.ln_ffn(x))


@dataclass(frozen=True)
class DecoderLayer:
    ln_self: LayerNorm
    self_attn: Attention
    ln_cross: LayerNorm
    cross_attn: Attention
    ln_ffn: LayerNorm
    ffn: FeedForward

    def __call__(
        self,
        x: np.ndarray,
        memory: np.ndarray,
        memory_mask: np.ndarray,
        self_mask: np.ndarray | None,
        start: int = 0,
        cache: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> np.ndarray:
        x = x + self.self_attn.self_attention(
            self.ln_self(x),
            self_mask,
            causal=cache is None,
            start=start,
            cache=None if cache is None else cache["self"],
        )
        x = x + self.cross_attn.cross_attention(
            self.ln_cross(x),
            memory,
            memory_mask,
            cache=None if cache is None else cache["cross"],
        )
        return x + self.ffn(self.ln_ffn(x))


@dataclass(frozen=True)
class DecodeResult:
    token_ids: list[int]
    mean_log_prob: float


class ModelBNumpy:
    """Inference-only Model B. Build with :meth:`load` from an exported ``.npz``."""

    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        meta = {k[len("meta.") :]: float(v) for k, v in w.items() if k.startswith("meta.")}
        self.d_model = int(meta["d_model"])
        self.n_heads = int(meta["n_heads"])
        self.d_head = self.d_model // self.n_heads
        self.eps = meta["ln_epsilon"]
        self.theta_base = meta["rope_theta_base"]
        self.enc_max_length = int(meta["encoder_max_length"])
        self.dec_max_length = int(meta["decoder_max_length"])
        self.use_rope = bool(meta["use_rope"])
        self.embedding = w["embedding"]
        self.output_bias = w["output_bias"]
        self.scale = np.float32(np.sqrt(self.d_model))

        enc_rope = self._rope(self.enc_max_length)
        dec_rope = self._rope(self.dec_max_length)
        self.encoder_layers = [
            EncoderLayer(
                ln_self=self._ln(w, f"encoder.{i}.ln_self"),
                self_attn=self._attn(w, f"encoder.{i}.self_attn", enc_rope),
                ln_ffn=self._ln(w, f"encoder.{i}.ln_ffn"),
                ffn=self._ffn(w, f"encoder.{i}.ffn"),
            )
            for i in range(int(meta["n_encoder_layers"]))
        ]
        self.encoder_final_ln = self._ln(w, "encoder.final_ln")
        self.decoder_layers = [
            DecoderLayer(
                ln_self=self._ln(w, f"decoder.{i}.ln_self"),
                self_attn=self._attn(w, f"decoder.{i}.self_attn", dec_rope),
                ln_cross=self._ln(w, f"decoder.{i}.ln_cross"),
                cross_attn=self._attn(w, f"decoder.{i}.cross_attn", None),
                ln_ffn=self._ln(w, f"decoder.{i}.ln_ffn"),
                ffn=self._ffn(w, f"decoder.{i}.ffn"),
            )
            for i in range(int(meta["n_decoder_layers"]))
        ]
        self.decoder_final_ln = self._ln(w, "decoder.final_ln")

    @classmethod
    def load(cls, path: str | Path) -> ModelBNumpy:
        with np.load(path) as data:
            return cls({key: data[key] for key in data.files})

    # -- builders -----------------------------------------------------------

    def _rope(self, max_length: int) -> RoPE | None:
        if not self.use_rope:
            return None
        return RoPE(max_length=max_length, d_head=self.d_head, theta_base=self.theta_base)

    def _ln(self, w: dict[str, np.ndarray], name: str) -> LayerNorm:
        return LayerNorm(w[f"{name}.gamma"], w[f"{name}.beta"], self.eps)

    @staticmethod
    def _dense(w: dict[str, np.ndarray], name: str) -> Dense:
        return Dense(w[f"{name}.kernel"], w[f"{name}.bias"])

    def _attn(self, w: dict[str, np.ndarray], name: str, rope: RoPE | None) -> Attention:
        return Attention(
            *(self._dense(w, f"{name}.{proj}") for proj in ("wq", "wk", "wv", "wo")),
            n_heads=self.n_heads,
            rope=rope,
        )

    def _ffn(self, w: dict[str, np.ndarray], name: str) -> FeedForward:
        return FeedForward(self._dense(w, f"{name}.inner"), self._dense(w, f"{name}.outer"))

    # -- forward ------------------------------------------------------------

    def embed(self, ids: np.ndarray) -> np.ndarray:
        return self.embedding[ids] * self.scale

    def encode(self, encoder_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(encoder_output[B, L, d_model], padding_mask[B, L])``."""
        encoder_ids = np.atleast_2d(np.asarray(encoder_ids, dtype=np.int64))
        mask = encoder_ids != PAD_ID
        x = self.embed(encoder_ids)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return self.encoder_final_ln(x), mask

    def logits(self, encoder_ids: np.ndarray, decoder_ids: np.ndarray) -> np.ndarray:
        """Teacher-forced logits ``[B, L_dec, vocab]`` (same as the Keras ``call``)."""
        memory, memory_mask = self.encode(encoder_ids)
        decoder_ids = np.atleast_2d(np.asarray(decoder_ids, dtype=np.int64))
        x = self.embed(decoder_ids)
        self_mask = decoder_ids != PAD_ID
        for layer in self.decoder_layers:
            x = layer(x, memory, memory_mask, self_mask)
        return self._project(self.decoder_final_ln(x))

    def _project(self, x: np.ndarray) -> np.ndarray:
        return x @ self.embedding.T + self.output_bias

    def greedy_decode(
        self,
        encoder_ids: list[int],
        max_length: int | None = None,
        max_repeat: int | None = 3,
    ) -> DecodeResult:
        """KV-cached greedy decoding; RoPE positions advance with each step.

        ``max_repeat`` stops a degenerate loop once the same token has been
        emitted that many times in a row (``None`` disables the guard).
        """
        max_length = min(max_length or self.dec_max_length, self.dec_max_length)
        memory, memory_mask = self.encode(np.asarray([encoder_ids]))
        caches = [{"self": {}, "cross": {}} for _ in self.decoder_layers]
        token = BOS_ID
        output: list[int] = []
        log_probs: list[float] = []
        for step in range(max_length):
            x = self.embed(np.asarray([[token]]))
            for layer, cache in zip(self.decoder_layers, caches, strict=True):
                x = layer(x, memory, memory_mask, None, start=step, cache=cache)
            logits = self._project(self.decoder_final_ln(x))[0, -1]
            shifted = logits - logits.max()
            log_softmax = shifted - np.log(np.exp(shifted).sum())
            token = int(np.argmax(logits))
            log_probs.append(float(log_softmax[token]))
            if token == EOS_ID:
                break
            if max_repeat and output[-max_repeat:] == [token] * max_repeat:
                break
            output.append(token)
        mean = float(np.mean(log_probs)) if log_probs else float("-inf")
        return DecodeResult(token_ids=output, mean_log_prob=mean)

    def pooled_encoding(self, encoder_ids: list[int]) -> np.ndarray:
        """Mean of encoder states over non-padding positions, L2-normalised."""
        memory, mask = self.encode(np.asarray([encoder_ids]))
        weights = mask[0].astype(np.float32)[:, None]
        pooled = (memory[0] * weights).sum(axis=0) / max(float(weights.sum()), 1.0)
        return pooled / (np.linalg.norm(pooled) + 1e-9)
