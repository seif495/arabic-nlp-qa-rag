### ~~~ GLOBAL IMPORT ~~~ ###
import numpy as np
import tensorflow as tf

### ~~~ LOCAL IMPORT ~~~ ###
from src.ms2.util import PipelineStep, get_config_value, load_pipeline_config

### ~~~ STATE MANAGEMENT ~~~ ###
# None


def sinusoidal_positional_encoding(max_length: int, d_model: int) -> tf.Tensor:
    """
    Build a vanilla sinusoidal positional-encoding lookup table.

    Args:
        max_length: Maximum sequence length supported by the table.
        d_model: Model dimension.

    Returns:
        Positional-encoding table of shape ``(1, max_length, d_model)``.
    """
    ### create position and dimension grids ###
    positions = np.arange(max_length)[:, np.newaxis]
    dimensions = np.arange(d_model)[np.newaxis, :]

    ### compute base transformer angles ###
    angle_rates = 1.0 / np.power(10000.0, (2 * (dimensions // 2)) / np.float32(d_model))
    angles = positions * angle_rates

    ### apply sin to even dims and cos to odd dims ###
    pe = np.zeros((max_length, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return tf.constant(pe[np.newaxis, ...], dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Add vanilla sinusoidal positional encodings to token embeddings.

    Input shape:
        ``(B, L, d_model)``

    Output shape:
        ``(B, L, d_model)``
    """

    def __init__(self, **kwargs: object) -> None:
        """
        Initialize the positional-encoding layer.

        Args:
            **kwargs: Additional keyword arguments passed to
                ``tf.keras.layers.Layer``.

        Returns:
            None.
        """
        ### init ###
        super().__init__(**kwargs)

        ### load config ###
        config = load_pipeline_config(PipelineStep.model_definition)

        ### get config values ###
        d_model: int = int(
            get_config_value(config, "transformer_embedding", "d_model", 192)
        )
        max_length: int = int(
            get_config_value(
                config, "transformer_positional_encoding", "max_length", 512
            )
        )
        dropout: float = float(
            get_config_value(config, "transformer_positional_encoding", "dropout", 0.1)
        )

        ### precompute positional encodings and dropout ###
        self.pe = sinusoidal_positional_encoding(max_length=max_length, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Add positional encodings and apply dropout.

        Args:
            x: Embedded token tensor of shape ``(B, L, d_model)``.
            training: Whether layer is running in training mode.

        Returns:
            Position-aware tensor of shape ``(B, L, d_model)``.
        """
        ### slice positional table to current sequence length ###
        seq_len = tf.shape(x)[1]
        pe_slice = tf.cast(self.pe[:, :seq_len, :], dtype=x.dtype)

        ### add positional signal to token embeddings ###
        # dim(output) = (B, L, d_model)
        output = x + pe_slice

        ### apply dropout ###
        output = self.dropout(output, training=training)

        return output
