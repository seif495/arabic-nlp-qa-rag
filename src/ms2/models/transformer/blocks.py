### ~~~ GLOBAL IMPORT ~~~ ###
import tensorflow as tf

### ~~~ LOCAL IMPORT ~~~ ###
from src.ms2.util import PipelineStep, get_config_value, load_pipeline_config

### ~~~ STATE MANAGEMENT ~~~ ###
# None


def make_padding_mask(token_ids: tf.Tensor) -> tf.Tensor:
    """
    Build a key-padding mask for attention layers.
    Args:
        token_ids: Token IDs of shape ``(B, L)`` with pad id ``0``.
    Returns:
        Mask of shape ``(B, 1, L)`` where True means keep.
    """
    mask = tf.not_equal(token_ids, 0)
    return mask[:, None, :]


def make_causal_mask(length: tf.Tensor) -> tf.Tensor:
    """
    Build a lower-triangular causal attention mask.
    Args:
        length: Decoder sequence length scalar tensor.
    Returns:
        Boolean mask of shape ``(1, L, L)``.
    """
    ones = tf.ones((length, length), dtype=tf.bool)
    lower = tf.linalg.band_part(ones, -1, 0)
    return lower[None, :, :]


class EncoderBlock(tf.keras.layers.Layer):
    """
    Vanilla transformer encoder block.

    Structure:
        LN -> MHA -> residual -> LN -> FFN -> residual
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        config = load_pipeline_config(PipelineStep.model_definition)

        d_model = int(get_config_value(config, "transformer_embedding", "d_model", 192))
        num_heads = int(get_config_value(config, "transformer_encoder", "num_heads", 4))
        d_ff = int(get_config_value(config, "transformer_encoder", "d_ff", 512))
        dropout_attn = float(
            get_config_value(config, "transformer_encoder", "dropout_attn", 0.1)
        )
        dropout_ff = float(
            get_config_value(config, "transformer_encoder", "dropout_ff", 0.1)
        )

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_attn,
            name="enc_mha",
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout_attn)

        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(d_ff, activation=tf.keras.activations.gelu),
                tf.keras.layers.Dropout(dropout_ff),
                tf.keras.layers.Dense(d_model),
            ],
            name="enc_ffn",
        )
        self.dropout2 = tf.keras.layers.Dropout(dropout_ff)

    def call(
        self, x: tf.Tensor, padding_mask: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        ### self-attention block ###
        h = self.norm1(x)
        attn_out = self.attn(
            query=h,
            value=h,
            key=h,
            attention_mask=padding_mask,
            training=training,
        )
        x = x + self.dropout1(attn_out, training=training)

        ### feed-forward block ###
        h = self.norm2(x)
        ffn_out = self.ffn(h, training=training)
        x = x + self.dropout2(ffn_out, training=training)

        return x


class DecoderBlock(tf.keras.layers.Layer):
    """
    Vanilla transformer decoder block.

    Structure:
        LN -> masked self-attn -> residual
        LN -> cross-attn -> residual
        LN -> FFN -> residual
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        config = load_pipeline_config(PipelineStep.model_definition)

        d_model = int(get_config_value(config, "transformer_embedding", "d_model", 192))
        num_heads = int(get_config_value(config, "transformer_decoder", "num_heads", 4))
        d_ff = int(get_config_value(config, "transformer_decoder", "d_ff", 512))
        dropout_attn = float(
            get_config_value(config, "transformer_decoder", "dropout_attn", 0.1)
        )
        dropout_ff = float(
            get_config_value(config, "transformer_decoder", "dropout_ff", 0.1)
        )

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_attn,
            name="dec_self_mha",
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout_attn)

        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_attn,
            name="dec_cross_mha",
        )
        self.dropout2 = tf.keras.layers.Dropout(dropout_attn)

        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(d_ff, activation=tf.keras.activations.gelu),
                tf.keras.layers.Dropout(dropout_ff),
                tf.keras.layers.Dense(d_model),
            ],
            name="dec_ffn",
        )
        self.dropout3 = tf.keras.layers.Dropout(dropout_ff)

    def call(
        self,
        x: tf.Tensor,
        memory: tf.Tensor,
        self_mask: tf.Tensor,
        memory_mask: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        ### masked self-attention ###
        h = self.norm1(x)
        self_attn = self.self_attn(
            query=h,
            value=h,
            key=h,
            attention_mask=self_mask,
            training=training,
        )
        x = x + self.dropout1(self_attn, training=training)

        ### cross-attention over encoder memory ###
        h = self.norm2(x)
        cross_attn = self.cross_attn(
            query=h,
            value=memory,
            key=memory,
            attention_mask=memory_mask,
            training=training,
        )
        x = x + self.dropout2(cross_attn, training=training)

        ### feed-forward ###
        h = self.norm3(x)
        ffn_out = self.ffn(h, training=training)
        x = x + self.dropout3(ffn_out, training=training)

        return x
