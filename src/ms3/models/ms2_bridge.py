import os
from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np

class MS2Bridge:
    """
    Unified adapter wrapping all three MS2 architectures for use in the RAG pipeline.
    
    Supports model_type in {"model_a", "model_b", "transformer"}.
    Auto-discovers checkpoints in experiments/ms2/.
    """
    def __init__(self, model_type: str, checkpoint_path: Optional[str] = None, repo_root: Optional[Path] = None):
        self.model_type = model_type.lower()
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._tokenizer = None
        self._tf = None

    @property
    def tf(self):
        if self._tf is None:
            import tensorflow as tf
            self._tf = tf
        return self._tf

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from src.ms2.data.tokenizer import ensure_default_tokenizer
            self._tokenizer = ensure_default_tokenizer(repo_root=self.repo_root)
        return self._tokenizer

    def _get_model_class(self):
        if self.model_type == "model_a":
            from src.ms2.models.rnn.model_a import ModelA
            return ModelA
        elif self.model_type == "model_b":
            from src.ms2.models.transformer.model_b import ModelB
            return ModelB
        elif self.model_type == "transformer":
            from src.ms2.models.transformer.model import TransformerModel
            return TransformerModel
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _load_model(self):
        model_class = self._get_model_class()
        model = model_class()
        
        # Build model by running a dummy input
        self._build_model(model)
        
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            model.load_weights(self.checkpoint_path)
        else:
            # Try to auto-discover
            discovered = self._discover_checkpoint()
            if discovered:
                print(f"Loading discovered checkpoint: {discovered}")
                model.load_weights(str(discovered))
            else:
                print(f"Warning: No checkpoint found for {self.model_type}. Using uninitialized weights.")
        
        return model

    def _build_model(self, model):
        from src.ms2.data.length_caps import LENGTH_CAPS
        tf = self.tf
        if self.model_type == "model_a":
            dummy_inputs = (
                tf.zeros((1, LENGTH_CAPS["l_q"]), dtype=tf.int32),
                tf.zeros((1, LENGTH_CAPS["l_c"]), dtype=tf.int32),
                tf.zeros((1, LENGTH_CAPS["l_enc"]), dtype=tf.int32),
                tf.zeros((1, LENGTH_CAPS["l_enc"], 16), dtype=tf.int32),
                tf.zeros((1, LENGTH_CAPS["l_dec"]), dtype=tf.int32),
            )
            model(dummy_inputs, training=False)
        elif self.model_type in ("model_b", "transformer"):
            dummy_inputs = (
                tf.zeros((1, LENGTH_CAPS["l_enc"]), dtype=tf.int32),
                tf.zeros((1, LENGTH_CAPS["l_dec"]), dtype=tf.int32),
            )
            model(dummy_inputs, training=False)

    def _discover_checkpoint(self) -> Optional[Path]:
        model_folder = self.model_type if self.model_type.startswith("model_") else f"model_{self.model_type}"
        base_dir = self.repo_root / "experiments" / "ms2" / model_folder
        if not base_dir.exists():
            return None
        
        # Look for best.weights.h5 in any seed dir
        for seed_dir in sorted(base_dir.iterdir()):
            if seed_dir.is_dir():
                ckpt = seed_dir / "checkpoints" / "best.weights.h5"
                if ckpt.exists():
                    return ckpt
        return None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def encode_for_reranking(self, question: str, chunks: List[str]) -> List[float]:
        """
        Encodes (question, chunk) pairs and returns similarity scores.
        """
        tf = self.tf
        tokenizer = self.tokenizer
        from src.ms2.data.length_caps import LENGTH_CAPS
        
        # Ensure model is loaded
        _ = self.model
        
        q_ids = tf.constant([_pad(tokenizer.encode(question), LENGTH_CAPS["l_q"])], dtype=tf.int32)

        scores = []
        for chunk in chunks:
            c_ids = tf.constant([_pad(tokenizer.encode(chunk), LENGTH_CAPS["l_c"])], dtype=tf.int32)
            
            if self.model_type == "model_a":
                b1_pooled, _ = self.model.branch1(q_ids, training=False)
                b2_seq, _ = self.model.branch2(c_ids, training=False)
                b2_pooled = tf.reduce_mean(b2_seq, axis=1)
                
                norm_q = tf.nn.l2_normalize(b1_pooled, axis=-1)
                norm_c = tf.nn.l2_normalize(b2_pooled, axis=-1)
                score = tf.reduce_sum(norm_q * norm_c, axis=-1)
                scores.append(float(score.numpy()[0]))
            
            elif self.model_type in ("model_b", "transformer"):
                if self.model_type == "model_b":
                    enc_out, _, _ = self.model.encoder(c_ids, training=False)
                    q_enc_out, _, _ = self.model.encoder(q_ids, training=False)
                else:
                    # TransformerModel returns it in details
                    q_res = self.model(q_ids, tf.zeros((1, 1), dtype=tf.int32), training=False, return_details=True)
                    q_enc_out = q_res["encoder_output"]
                    c_res = self.model(c_ids, tf.zeros((1, 1), dtype=tf.int32), training=False, return_details=True)
                    enc_out = c_res["encoder_output"]
                
                q_pooled = tf.reduce_mean(q_enc_out, axis=1)
                c_pooled = tf.reduce_mean(enc_out, axis=1)
                
                norm_q = tf.nn.l2_normalize(q_pooled, axis=-1)
                norm_c = tf.nn.l2_normalize(c_pooled, axis=-1)
                score = tf.reduce_sum(norm_q * norm_c, axis=-1)
                scores.append(float(score.numpy()[0]))
        
        return scores

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generates an answer greedily.
        """
        tf = self.tf
        tokenizer = self.tokenizer
        from src.ms2.data.length_caps import LENGTH_CAPS
        from src.ms2.data.tokenizer import BOS_ID, EOS_ID, SEP_ID
        
        # Ensure model is loaded
        _ = self.model
        
        q_tokens = tokenizer.encode(question)[:LENGTH_CAPS["l_q"]]
        c_tokens = tokenizer.encode(context)[:LENGTH_CAPS["l_c"]]
        
        if self.model_type == "model_a":
            from src.ms2.data.pipeline import _pad_char_matrix, _encoder_pieces
            q_ids = tf.constant([_pad(q_tokens, LENGTH_CAPS["l_q"])], dtype=tf.int32)
            c_ids = tf.constant([_pad(c_tokens, LENGTH_CAPS["l_c"])], dtype=tf.int32)
            joint_ids = tf.constant([_pad([BOS_ID, *q_tokens, SEP_ID, *c_tokens, EOS_ID], LENGTH_CAPS["l_enc"])], dtype=tf.int32)
            pieces = _encoder_pieces(question, context, tokenizer)
            char_ids = tf.constant([_pad_char_matrix(pieces, tokenizer, LENGTH_CAPS["l_enc"])], dtype=tf.int32)
            
            res = self.model((q_ids, c_ids, joint_ids, char_ids, tf.constant([[BOS_ID]], dtype=tf.int32)), training=False)
            encoder_summary = res["encoder_summary"]
            gamma = res["gamma"]
            beta = res["beta"]
            
            output_ids = [BOS_ID]
            for _ in range(LENGTH_CAPS["l_dec"]):
                dec_in = tf.constant([output_ids], dtype=tf.int32)
                logits = self.model.decoder(dec_in, encoder_summary, gamma, beta, training=False)
                next_id = int(tf.argmax(logits[0, -1, :], axis=-1).numpy())
                if next_id == EOS_ID:
                    break
                output_ids.append(next_id)
            
            return tokenizer.decode(output_ids)

        elif self.model_type in ("model_b", "transformer"):
            enc_in = tf.constant([_pad([BOS_ID, *q_tokens, SEP_ID, *c_tokens, EOS_ID], LENGTH_CAPS["l_enc"])], dtype=tf.int32)
            output_ids = [BOS_ID]
            
            for _ in range(LENGTH_CAPS["l_dec"]):
                dec_in = tf.constant([output_ids], dtype=tf.int32)
                if self.model_type == "model_b":
                    res = self.model((enc_in, dec_in), training=False)
                    logits = res["logits"]
                else:
                    logits = self.model(enc_in, dec_in, training=False)
                
                next_id = int(tf.argmax(logits[0, -1, :], axis=-1).numpy())
                if next_id == EOS_ID:
                    break
                output_ids.append(next_id)
            
            return tokenizer.decode(output_ids)
            
        return ""

def _pad(values: List[int], target_length: int) -> List[int]:
    return values[:target_length] + [0] * max(0, target_length - len(values))
