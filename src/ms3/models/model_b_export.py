"""Export MS2 Model B (RoPE Transformer) weights for the NumPy web runtime.

The Vercel deployment (``web/``) cannot ship TensorFlow, so the trained Keras
weights are flattened into a single ``.npz`` that ``web/ms3web/model_b.py``
loads. Keys follow ``<stack>.<layer>.<block>.<param>`` naming; ``meta.*``
scalars carry the architecture constants the NumPy port needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def model_b_weight_dict(model: Any, dtype: str = "float16") -> dict[str, np.ndarray]:
    """Flatten a built ``ModelB`` into ``{name: array}``."""
    from src.ms2.models.transformer.attention import N_HEADS
    from src.ms2.models.transformer.embedding import D_MODEL
    from src.ms2.models.transformer.rope import THETA_BASE

    encoder = model.encoder
    decoder = model.decoder
    if encoder.layers_[0] is encoder.layers_[-1] and len(encoder.layers_) > 1:
        raise ValueError("shared_layers ablation is not supported by the web export")

    out: dict[str, np.ndarray] = {
        "embedding": model.embedding.embeddings.numpy(),
        "output_bias": model.output_bias.numpy(),
    }

    def dense(name: str, layer: Any) -> None:
        out[f"{name}.kernel"] = layer.kernel.numpy()
        out[f"{name}.bias"] = layer.bias.numpy()

    def norm(name: str, layer: Any) -> None:
        out[f"{name}.gamma"] = layer.gamma.numpy()
        out[f"{name}.beta"] = layer.beta.numpy()

    def attn(name: str, layer: Any) -> None:
        for proj in ("wq", "wk", "wv", "wo"):
            dense(f"{name}.{proj}", getattr(layer, proj))

    def ffn(name: str, block: Any) -> None:
        inner, outer = block.layers
        dense(f"{name}.inner", inner)
        dense(f"{name}.outer", outer)

    for idx, layer in enumerate(encoder.layers_):
        prefix = f"encoder.{idx}"
        norm(f"{prefix}.ln_self", layer.ln1)
        attn(f"{prefix}.self_attn", layer.self_attn)
        norm(f"{prefix}.ln_ffn", layer.ln2)
        ffn(f"{prefix}.ffn", layer.ffn)
    norm("encoder.final_ln", encoder.final_ln)

    for idx, layer in enumerate(decoder.layers_):
        prefix = f"decoder.{idx}"
        norm(f"{prefix}.ln_self", layer.ln_self)
        attn(f"{prefix}.self_attn", layer.self_attn)
        norm(f"{prefix}.ln_cross", layer.ln_cross)
        attn(f"{prefix}.cross_attn", layer.cross_attn)
        norm(f"{prefix}.ln_ffn", layer.ln_ffn)
        ffn(f"{prefix}.ffn", layer.ffn)
    norm("decoder.final_ln", decoder.final_ln)

    out = {key: value.astype(dtype) for key, value in out.items()}

    enc_rope = encoder.layers_[0].self_attn.rope
    dec_rope = decoder.layers_[0].self_attn.rope
    meta = {
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_encoder_layers": len(encoder.layers_),
        "n_decoder_layers": len(decoder.layers_),
        "ln_epsilon": encoder.final_ln.epsilon,
        "use_rope": int(encoder.positional_mode == "rope"),
        "rope_theta_base": THETA_BASE,
        "encoder_max_length": enc_rope.max_length if enc_rope is not None else 420,
        "decoder_max_length": dec_rope.max_length if dec_rope is not None else 64,
    }
    if encoder.positional_mode not in ("rope", "no_pe"):
        raise ValueError(
            f"positional_mode={encoder.positional_mode!r} is not supported by the web export"
        )
    for key, value in meta.items():
        out[f"meta.{key}"] = np.asarray(value, dtype=np.float64)
    return out


def build_model_b(checkpoint_path: Path | None = None) -> Any:
    """Instantiate ``ModelB``, build its variables, and optionally load weights."""
    import tensorflow as tf

    from src.ms2.data.length_caps import LENGTH_CAPS
    from src.ms2.models.transformer.model_b import ModelB

    model = ModelB()
    model(
        (
            tf.zeros((1, LENGTH_CAPS["l_enc"]), dtype=tf.int32),
            tf.zeros((1, LENGTH_CAPS["l_dec"]), dtype=tf.int32),
        ),
        training=False,
    )
    if checkpoint_path is not None:
        model.load_weights(str(checkpoint_path))
    return model


def export_model_b_npz(
    checkpoint_path: Path, output_path: Path, dtype: str = "float16"
) -> Path:
    model = build_model_b(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **model_b_weight_dict(model, dtype=dtype))
    return output_path
