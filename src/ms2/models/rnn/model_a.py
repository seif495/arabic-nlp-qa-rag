from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tensorflow as tf

from src.ms2.models.rnn.align import BranchAligner
from src.ms2.models.rnn.branch1 import Branch1QuestionEncoder
from src.ms2.models.rnn.branch2 import Branch2ContextEncoder
from src.ms2.models.rnn.branch3 import Branch3JointEncoder
from src.ms2.models.rnn.decoder import FiLMLSTMDecoder
from src.ms2.models.rnn.embedding import SharedTokenEmbedding
from src.ms2.models.rnn.film_generator import FiLMGenerator
from src.ms2.models.rnn.gated_merge import GatedMerge
from src.ms2.models.rnn.refinement import RefinementBiGRU


class ModelA(tf.keras.Model):
    """Tri-encoder gated-fusion seq2seq with FiLM decoder, ADR §2."""

    def __init__(self, char_vocab_size: int = 192, no_film: bool = False, mean_merge: bool = False, plain_branch3: bool = False, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.shared_embedding = SharedTokenEmbedding(name="model_a_shared_embedding")
        self.branch1 = Branch1QuestionEncoder(self.shared_embedding, name="branch1")
        self.branch2 = Branch2ContextEncoder(self.shared_embedding, name="branch2")
        self.branch3 = Branch3JointEncoder(self.shared_embedding, char_vocab_size=char_vocab_size, use_char_cnn=not plain_branch3, name="branch3")
        self.aligner = BranchAligner(name="branch_aligner")
        self.merge = GatedMerge(mean_merge=mean_merge, name="gated_merge")
        self.refinement = RefinementBiGRU(name="refinement")
        self.film = FiLMGenerator(name="film_generator")
        self.decoder = FiLMLSTMDecoder(self.shared_embedding, use_film=not no_film, name="decoder")

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], training: bool = False) -> dict[str, tf.Tensor]:
        question_ids, context_ids, joint_ids, char_ids, decoder_inputs = inputs
        question_lengths = tf.reduce_sum(tf.cast(tf.not_equal(question_ids, 0), tf.int32), axis=1)
        b1_pooled, _ = self.branch1(question_ids, training=training)
        b2_seq, context_mask = self.branch2(context_ids, training=training)
        b3_seq, _ = self.branch3(joint_ids, char_ids, training=training)
        aligned = self.aligner(b1_pooled, b2_seq, b3_seq, question_lengths)
        fused, gates = self.merge(aligned, training=training)
        encoder_output, encoder_summary = self.refinement(fused, mask=context_mask, training=training)
        gamma, beta = self.film(encoder_summary)
        logits = self.decoder(decoder_inputs, encoder_summary, gamma, beta, training=training)
        return {
            "logits": logits,
            "gates": gates,
            "gamma": gamma,
            "beta": beta,
            "encoder_summary": encoder_summary,
            "encoder_output": encoder_output,
        }


def parameter_audit(model: ModelA) -> dict[str, Any]:
    total = int(model.count_params())
    return {
        "model": "A",
        "total_parameters": total,
        "budget_min": 2_400_000,
        "budget_max": 3_000_000,
        "within_budget": 2_400_000 <= total <= 3_000_000,
        "note": "Exact count depends on concrete char vocab size; architecture follows ADR §2.14 categories.",
    }


def write_parameter_audit(model: ModelA, path: Path) -> dict[str, Any]:
    payload = parameter_audit(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
