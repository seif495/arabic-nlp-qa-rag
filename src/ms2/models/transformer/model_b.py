from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tensorflow as tf

from src.ms2.models.transformer.decoder import TransformerDecoder
from src.ms2.models.transformer.embedding import VOCAB_SIZE, TransformerTokenEmbedding
from src.ms2.models.transformer.encoder import TransformerEncoder
from src.ms2.models.transformer.kv_cache import KVCache


class ModelB(tf.keras.Model):
    """Small RoPE encoder-decoder Transformer from ADR §3."""

    def __init__(self, positional_mode: str = "rope", shared_layers: bool = False, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.embedding = TransformerTokenEmbedding(name="model_b_shared_embedding")
        self.encoder = TransformerEncoder(self.embedding, positional_mode=positional_mode, shared_layers=shared_layers, name="transformer_encoder")
        self.decoder = TransformerDecoder(self.embedding, positional_mode=positional_mode, shared_layers=shared_layers, name="transformer_decoder")
        self.output_bias = self.add_weight(name="model_b_output_bias", shape=(VOCAB_SIZE,), initializer="zeros")

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False, cache: KVCache | None = None) -> dict[str, tf.Tensor | list[tf.Tensor]]:
        encoder_ids, decoder_ids = inputs
        encoder_output, enc_attn, enc_mask = self.encoder(encoder_ids, training=training)
        decoder_output, _, cross_attn = self.decoder(decoder_ids, encoder_output, enc_mask, training=training, cache=cache)
        logits = tf.matmul(decoder_output, self.embedding.embeddings, transpose_b=True) + tf.cast(self.output_bias, decoder_output.dtype)
        return {"logits": logits, "encoder_self_attn_weights": enc_attn, "decoder_cross_attn_weights": cross_attn}

    def init_cache(self) -> KVCache:
        return KVCache.for_layers(len(self.decoder.layers_))


def parameter_audit(model: ModelB) -> dict[str, Any]:
    total = int(model.count_params())
    return {"model": "B", "total_parameters": total, "budget_min": 3_400_000, "budget_max": 4_200_000, "within_budget": 3_400_000 <= total <= 4_200_000}


def write_parameter_audit(model: ModelB, path: Path) -> dict[str, Any]:
    payload = parameter_audit(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
