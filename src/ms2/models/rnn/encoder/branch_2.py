### ~~~ GLOBAL IMPORT ~~~ ###
import tensorflow as tf

### ~~~ LOCAL IMPORT ~~~ ###
from src.ms2.util import (
    load_pipeline_config,
    PipelineStep,
    get_config_value,
)
from src.ms2.models.rnn.embedder import Embedder

### ~~~ STATE MANAGEMENT ~~~ ###
# None


class Branch2(tf.keras.layers.Layer):
    """
    Context encoder branch for the RNN-based model.
    Branch2 embeds a sequence of context token IDs, applies embedding dropout, and
    encodes the sequence using two stacked bidirectional GRU layers. Padding tokens
    with ID ``0`` are masked so they do not contribute to the recurrent updates.
    This branch preserves the full temporal sequence instead of pooling it, producing
    one contextual representation per input token.
    Input shape:
        ``(B, L)``
    Output shape:
        ``(B, L, 2 * output_dim)``
    """

    def __init__(self, embedder: Embedder, **kwargs: object) -> None:
        """
        Initialize the Branch2 context encoder.
        Loads the branch-specific hyperparameters from the model-definition config,
        stores the shared token embedder, and constructs the embedding dropout layer,
        two bidirectional GRU layers, and the intermediate recurrent dropout layer.
        Args:
            embedder: Shared token embedding layer used to map token IDs to dense
                vectors.
            **kwargs: Additional keyword arguments passed to ``tf.keras.layers.Layer``.
        Returns:
            None.
        """
        ### init ###
        super().__init__(**kwargs)

        ### load up the config ###
        config = load_pipeline_config(PipelineStep.model_definition)

        ### get all the config values ###
        dropout_embedding: float = get_config_value(
            config, "rnn_branch_2", "dropout_embedding", 0.1
        )
        dropout_rnn: float = get_config_value(
            config, "rnn_branch_2", "dropout_rnn", 0.2
        )
        output_dim: int = get_config_value(config, "rnn_branch_2", "output_dim", 128)

        ### define the layers ###
        self.embedder = embedder
        self.dropout_embedding = tf.keras.layers.Dropout(dropout_embedding)
        self.gru1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                output_dim,
                return_sequences=True,
                dropout=dropout_rnn,
            ),
            name="branch2_gru1",
        )
        self.dropout_rnn = tf.keras.layers.Dropout(dropout_rnn)
        self.gru2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                output_dim,
                return_sequences=True,
                dropout=dropout_rnn,
            ),
            name="branch2_gru2",
        )

    def call(self, token_ids: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Encode a batch of token sequences with stacked bidirectional GRUs.
        Creates a padding mask from ``token_ids != 0``, embeds the token IDs, applies
        embedding dropout, passes the sequence through the first Bi-GRU, applies
        recurrent dropout, and then passes the result through the second Bi-GRU.
        Args:
            token_ids: Integer token ID tensor of shape ``(B, L)``, where ``B`` is the
                batch size and ``L`` is the sequence length. Token ID ``0`` is treated
                as padding.
            training: Whether the layer is running in training mode. Controls dropout
                behavior.
        Returns:
            A contextual sequence tensor of shape ``(B, L, 2 * output_dim)``.
        """
        ### define the mask ###
        # 0 here is for padding tokens
        mask: tf.Tensor = tf.not_equal(token_ids, 0)

        ### get the token embeddings ###
        # dim(x) = (B, L, D_embed)
        x = self.dropout_embedding(self.embedder(token_ids), training=training)

        ### encode the sequence with the first bidirectional GRU ###
        # dim(seq) = (B, L, 2*D_out) because it's bidirectional
        seq = self.gru1(x, mask=mask, training=training)

        ### apply dropout to the sequence ###
        seq = self.dropout_rnn(seq, training=training)

        ### encode the sequence with the second bidirectional GRU ###
        # dim(seq) = (B, L, 2*D_out) because it's bidirectional
        seq = self.gru2(seq, mask=mask, training=training)

        return seq


if __name__ == "__main__":
    pass
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/
