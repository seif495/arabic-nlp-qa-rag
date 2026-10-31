"""Export MS2 model weights for the NumPy web runtime.

The Vercel deployment (``web/``) cannot ship TensorFlow, so trained Keras
weights are flattened into one ``.npz``. Keys follow
``<stack>.<layer>.<block>.<param>`` naming; ``meta.*`` scalars carry the
architecture switches the NumPy engines need.

Supported candidates (see ``src/ms3/models/ms2_web_selection.py``):

- ``model_a`` (tri-encoder RNN + FiLM) -> ``web/ms3web/rnn.py``;
- ``model_b`` (RoPE Transformer) and ``transformer`` (vanilla sinusoidal)
  -> ``web/ms3web/seq2seq.py``.

``rnn_model`` has no web runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

EXPORTABLE = ("model_a", "model_b", "transformer")
RUNTIMES = {"model_a": "model_a_rnn", "model_b": "seq2seq_transformer", "transformer": "seq2seq_transformer"}
POSITIONAL_MODE_IDS = {"none": 0, "no_pe": 0, "rope": 1, "sinusoidal": 2}


class _Writer:
    def __init__(self) -> None:
        self.out: dict[str, np.ndarray] = {}

    def array(self, name: str, value: Any) -> None:
        self.out[name] = np.asarray(value.numpy() if hasattr(value, "numpy") else value)

    def dense(self, name: str, layer: Any) -> None:
        self.array(f"{name}.kernel", layer.kernel)
        self.array(f"{name}.bias", layer.bias)

    def norm(self, name: str, layer: Any) -> None:
        self.array(f"{name}.gamma", layer.gamma)
        self.array(f"{name}.beta", layer.beta)

    def projections(self, name: str, layer: Any) -> None:
        for proj in ("wq", "wk", "wv", "wo"):
            self.dense(f"{name}.{proj}", getattr(layer, proj))

    def keras_mha(self, name: str, mha: Any) -> None:
        """``tf.keras.layers.MultiHeadAttention`` -> flat ``(d, h*k)`` projections."""
        for proj, dense in (("wq", mha._query_dense), ("wk", mha._key_dense), ("wv", mha._value_dense)):
            kernel = dense.kernel.numpy()  # (d_model, heads, key_dim)
            self.array(f"{name}.{proj}.kernel", kernel.reshape(kernel.shape[0], -1))
            self.array(f"{name}.{proj}.bias", dense.bias.numpy().reshape(-1))
        kernel = mha._output_dense.kernel.numpy()  # (heads, key_dim, d_model)
        self.array(f"{name}.wo.kernel", kernel.reshape(-1, kernel.shape[-1]))
        self.array(f"{name}.wo.bias", mha._output_dense.bias)

    def ffn(self, name: str, block: Any) -> None:
        inner, *_, outer = [layer for layer in block.layers if hasattr(layer, "kernel")]
        self.dense(f"{name}.inner", inner)
        self.dense(f"{name}.outer", outer)

    def rnn_cell(self, name: str, layer: Any) -> None:
        """Keras LSTM/GRU layer (or its cell) -> kernel / recurrent_kernel / bias."""
        cell = getattr(layer, "cell", layer)
        self.array(f"{name}.kernel", cell.kernel)
        self.array(f"{name}.recurrent_kernel", cell.recurrent_kernel)
        self.array(f"{name}.bias", cell.bias)

    def bidirectional(self, name: str, layer: Any) -> None:
        self.rnn_cell(f"{name}.forward", layer.forward_layer)
        self.rnn_cell(f"{name}.backward", layer.backward_layer)

    def meta(self, **values: float) -> None:
        for key, value in values.items():
            self.out[f"meta.{key}"] = np.asarray(value, dtype=np.float64)


def model_b_weight_dict(model: Any, dtype: str = "float16") -> dict[str, np.ndarray]:
    """Flatten a built ``ModelB`` (RoPE Transformer)."""
    from src.ms2.models.transformer.attention import N_HEADS
    from src.ms2.models.transformer.embedding import D_MODEL
    from src.ms2.models.transformer.rope import THETA_BASE

    encoder, decoder = model.encoder, model.decoder
    if len(encoder.layers_) > 1 and encoder.layers_[0] is encoder.layers_[-1]:
        raise ValueError("shared_layers ablation is not supported by the web export")
    if encoder.positional_mode not in ("rope", "no_pe"):
        raise ValueError(f"positional_mode={encoder.positional_mode!r} is not supported by the web export")

    w = _Writer()
    w.array("embedding", model.embedding.embeddings)
    w.array("output_bias", model.output_bias)
    for idx, layer in enumerate(encoder.layers_):
        prefix = f"encoder.{idx}"
        w.norm(f"{prefix}.ln_self", layer.ln1)
        w.projections(f"{prefix}.self_attn", layer.self_attn)
        w.norm(f"{prefix}.ln_ffn", layer.ln2)
        w.ffn(f"{prefix}.ffn", layer.ffn)
    w.norm("encoder.final_ln", encoder.final_ln)
    for idx, layer in enumerate(decoder.layers_):
        prefix = f"decoder.{idx}"
        w.norm(f"{prefix}.ln_self", layer.ln_self)
        w.projections(f"{prefix}.self_attn", layer.self_attn)
        w.norm(f"{prefix}.ln_cross", layer.ln_cross)
        w.projections(f"{prefix}.cross_attn", layer.cross_attn)
        w.norm(f"{prefix}.ln_ffn", layer.ln_ffn)
        w.ffn(f"{prefix}.ffn", layer.ffn)
    w.norm("decoder.final_ln", decoder.final_ln)

    out = {key: value.astype(dtype) for key, value in w.out.items()}
    enc_rope = encoder.layers_[0].self_attn.rope
    dec_rope = decoder.layers_[0].self_attn.rope
    w.out = out
    w.meta(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_encoder_layers=len(encoder.layers_),
        n_decoder_layers=len(decoder.layers_),
        ln_epsilon=encoder.final_ln.epsilon,
        positional_mode=POSITIONAL_MODE_IDS[encoder.positional_mode],
        rope_theta_base=THETA_BASE,
        encoder_max_length=enc_rope.max_length if enc_rope is not None else 420,
        decoder_max_length=dec_rope.max_length if dec_rope is not None else 64,
        encoder_final_ln=1,
        tied_output=1,
    )
    return w.out


def vanilla_transformer_weight_dict(model: Any, dtype: str = "float16") -> dict[str, np.ndarray]:
    """Flatten a built vanilla ``TransformerModel`` (sinusoidal PE, Keras MHA)."""
    from src.ms2.data.length_caps import LENGTH_CAPS

    w = _Writer()
    w.array("embedding", model.token_embedding.embedding.embeddings)
    for idx, block in enumerate(model.encoder_blocks):
        prefix = f"encoder.{idx}"
        w.norm(f"{prefix}.ln_self", block.norm1)
        w.keras_mha(f"{prefix}.self_attn", block.attn)
        w.norm(f"{prefix}.ln_ffn", block.norm2)
        w.ffn(f"{prefix}.ffn", block.ffn)
    for idx, block in enumerate(model.decoder_blocks):
        prefix = f"decoder.{idx}"
        w.norm(f"{prefix}.ln_self", block.norm1)
        w.keras_mha(f"{prefix}.self_attn", block.self_attn)
        w.norm(f"{prefix}.ln_cross", block.norm2)
        w.keras_mha(f"{prefix}.cross_attn", block.cross_attn)
        w.norm(f"{prefix}.ln_ffn", block.norm3)
        w.ffn(f"{prefix}.ffn", block.ffn)
    w.norm("decoder.final_ln", model.final_norm)
    w.dense("output_projection", model.output_projection)

    w.out = {key: value.astype(dtype) for key, value in w.out.items()}
    mha = model.encoder_blocks[0].attn
    w.meta(
        d_model=model.token_embedding.d_model,
        n_heads=mha._num_heads,
        n_encoder_layers=len(model.encoder_blocks),
        n_decoder_layers=len(model.decoder_blocks),
        ln_epsilon=model.final_norm.epsilon,
        positional_mode=POSITIONAL_MODE_IDS["sinusoidal"],
        sinusoidal_max_length=int(model.positional_encoding.pe.shape[1]),
        encoder_max_length=int(model.positional_encoding.pe.shape[1]),
        decoder_max_length=LENGTH_CAPS["l_dec"],
        encoder_final_ln=0,
        tied_output=0,
    )
    return w.out


def model_a_weight_dict(model: Any, dtype: str = "float16") -> dict[str, np.ndarray]:
    """Flatten a built ``ModelA`` (tri-encoder RNN with FiLM decoder)."""
    from src.ms2.data.length_caps import LENGTH_CAPS

    if model.merge.mean_merge or not model.branch3.use_char_cnn:
        raise ValueError("Model A ablations are not supported by the web export")
    w = _Writer()
    w.array("embedding", model.shared_embedding.embeddings)
    w.array("char_embedding", model.branch3.char_embedding.embedding.embeddings)
    for idx, conv in enumerate(model.branch3.char_cnn.convs):
        w.dense(f"char_conv.{idx}", conv)
    w.bidirectional("branch1", model.branch1.encoder)
    w.dense("branch1.pool_proj", model.branch1.pool.proj)
    w.array("branch1.pool_score.kernel", model.branch1.pool.score.kernel)
    w.bidirectional("branch2.gru1", model.branch2.gru1)
    w.bidirectional("branch2.gru2", model.branch2.gru2)
    w.bidirectional("branch3", model.branch3.encoder)
    for idx, proj in enumerate((model.merge.proj1, model.merge.proj2, model.merge.proj3), start=1):
        w.dense(f"merge.proj{idx}", proj)
    w.dense("merge.gate_hidden", model.merge.gate_hidden)
    w.dense("merge.gate_out", model.merge.gate_out)
    w.norm("merge.norm", model.merge.norm)
    w.bidirectional("refine", model.refinement.bigru)
    w.dense("film.hidden", model.film.hidden)
    w.dense("film.out", model.film.out)
    decoder = model.decoder
    for idx in range(2):
        w.dense(f"decoder.h_init_{idx}", decoder.h_init[idx])
        w.dense(f"decoder.c_init_{idx}", decoder.c_init[idx])
    w.rnn_cell("decoder.lstm1", decoder.lstm1)
    w.rnn_cell("decoder.lstm2", decoder.lstm2)
    w.array("decoder.to_tok.kernel", decoder.output_projection.to_tok.kernel)
    w.array("decoder.output_bias", decoder.output_projection.bias)

    w.out = {key: value.astype(dtype) for key, value in w.out.items()}
    w.meta(
        ln_epsilon=model.merge.norm.epsilon,
        use_film=int(decoder.use_film),
        n_char_convs=len(model.branch3.char_cnn.convs),
        decoder_max_length=LENGTH_CAPS["l_dec"],
    )
    return w.out


def weight_dict(name: str, model: Any, dtype: str = "float16") -> dict[str, np.ndarray]:
    if name == "model_a":
        return model_a_weight_dict(model, dtype=dtype)
    if name == "model_b":
        return model_b_weight_dict(model, dtype=dtype)
    if name == "transformer":
        return vanilla_transformer_weight_dict(model, dtype=dtype)
    raise ValueError(f"{name!r} has no NumPy web runtime; exportable: {EXPORTABLE}")


def build_model(name: str, checkpoint_path: Path | None = None, repo_root: Path | None = None) -> Any:
    """Instantiate a candidate, build its variables, and optionally load weights."""
    import tensorflow as tf

    from src.ms2.data.length_caps import LENGTH_CAPS
    from src.ms3.models.ms2_web_selection import build_candidate

    model, forward = build_candidate(name, (repo_root or Path.cwd()).resolve())
    forward(
        model,
        {
            "encoder_input_ids": tf.zeros((1, LENGTH_CAPS["l_enc"]), dtype=tf.int32),
            "decoder_input_ids": tf.zeros((1, LENGTH_CAPS["l_dec"]), dtype=tf.int32),
            "question_ids": tf.zeros((1, LENGTH_CAPS["l_q"]), dtype=tf.int32),
            "context_ids": tf.zeros((1, LENGTH_CAPS["l_c"]), dtype=tf.int32),
            "char_matrix": tf.zeros((1, LENGTH_CAPS["l_enc"], 16), dtype=tf.int32),
        },
        False,
    )
    if checkpoint_path is not None:
        model.load_weights(str(checkpoint_path))
    return model


def export_npz(name: str, checkpoint_path: Path, output_path: Path, dtype: str = "float16") -> Path:
    model = build_model(name, checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **weight_dict(name, model, dtype=dtype))
    return output_path
