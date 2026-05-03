### ~~~ GLOBAL IMPORT ~~~ ###
import math
import tensorflow as tf

### ~~~ LOCAL IMPORT ~~~ ###
from src.ms2.models.rnn.embedder import get_vocab_size
from src.ms2.util import PipelineStep, get_config_value, load_pipeline_config

### ~~~ STATE MANAGEMENT ~~~ ###
# None


class TokenEmbedding(tf.keras.layers.Layer):
    """
    Token embedding layer for the vanilla Transformer (Model B).

    This layer maps integer token IDs to dense vectors and then scales the
    embeddings by ``sqrt(d_model)``. The scaling follows the standard
    Transformer convention and keeps activation magnitudes in a stable range.

    Input shape:
        ``(B, L)``

    Output shape:
        ``(B, L, d_model)``
    """

    def __init__(self, vocab_size: int | None = None, **kwargs: object) -> None:
        """
        Initialize the token embedding layer.

        Args:
            vocab_size: Optional explicit vocabulary size override. If omitted,
                vocab size is derived from the preprocessed vocab file.
            **kwargs: Additional keyword arguments passed to
                ``tf.keras.layers.Layer``.

        Returns:
            None.
        """
        ### init ###
        super().__init__(**kwargs)

        ### load model-definition config ###
        config = load_pipeline_config(PipelineStep.model_definition)

        ### get config values ###
        d_model: int = int(
            get_config_value(config, "transformer_embedding", "d_model", 192)
        )
        resolved_vocab_size: int = (
            get_vocab_size() if vocab_size is None else vocab_size
        )

        ### store dimension needed at call time ###
        self.d_model = d_model

        ### define the token embedding table ###
        self.embedding = tf.keras.layers.Embedding(
            input_dim=resolved_vocab_size,
            output_dim=d_model,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                stddev=d_model**-0.5
            ),
            mask_zero=True,
            name="transformer_token_embedding",
        )

    def call(self, token_ids: tf.Tensor) -> tf.Tensor:
        """
        Embed token IDs and apply ``sqrt(d_model)`` scaling.

        Args:
            token_ids: Integer token ID tensor of shape ``(B, L)``.

        Returns:
            Embedded tensor of shape ``(B, L, d_model)``.
        """
        ### get the raw token embeddings ###
        # dim(x) = (B, L, d_model)
        x = self.embedding(token_ids)

        ### apply standard transformer scaling ###
        # dim(output) = (B, L, d_model)
        output = x * tf.cast(math.sqrt(self.d_model), dtype=x.dtype)

        return output
