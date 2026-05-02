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


class TokenCNN(tf.keras.layers.Layer):
    """
    Token-level CNN encoder for extracting local n-gram features over token
    embeddings.
    TokenCNN applies multiple 1D convolution layers with different kernel sizes
    over the token sequence, concatenates their outputs, and returns a contextual
    token-level feature sequence. Padding is kept as ``same`` so the output length
    matches the input sequence length.
    Input shape:
        ``(B, L, D_embed)``
    Output shape:
        ``(B, L, output_dim)``
    """

    def __init__(self, **kwargs: object) -> None:
        """
        Initialize the token-level CNN encoder.
        Loads the token-CNN hyperparameters from the model-definition config and
        constructs a set of parallel Conv1D layers. The total output dimension is
        split across the configured kernel sizes.
        Args:
            **kwargs: Additional keyword arguments passed to
                ``tf.keras.layers.Layer``.
        Returns:
            None.
        """
        ### init ###
        super().__init__(**kwargs)

        ### load up the config ###
        config = load_pipeline_config(PipelineStep.model_definition)

        ### get all the config values ###
        output_dim: int = get_config_value(config, "rnn_branch_3", "token_cnn_dim", 64)
        kernel_sizes: tuple[int, ...] = tuple(
            get_config_value(
                config, "rnn_branch_3", "token_cnn_kernel_sizes", (2, 3, 4)
            )
        )
        activation: str = get_config_value(
            config, "rnn_branch_3", "token_cnn_activation", "relu"
        )

        ### split the filters across the kernel sizes ###
        base_filters: int = output_dim // len(kernel_sizes)
        filters: list[int] = [base_filters for _ in kernel_sizes]
        filters[-1] += output_dim - sum(filters)

        ### define the temporal convolution layers ###
        self.convs = [
            tf.keras.layers.Conv1D(
                filters=count,
                kernel_size=kernel,
                activation=activation,
                padding="same",
                name=f"token_conv_k{kernel}",
            )
            for kernel, count in zip(kernel_sizes, filters, strict=True)
        ]

    def call(self, token_embeddings: tf.Tensor) -> tf.Tensor:
        """
        Apply parallel token-level convolutions to a token embedding sequence.
        Each convolution scans over the sequence length dimension and extracts
        local token n-gram features. The outputs from all kernel sizes are
        concatenated along the feature dimension.
        Args:
            token_embeddings: Embedded token sequence of shape ``(B, L, D_embed)``,
                where ``B`` is the batch size, ``L`` is the token sequence length,
                and ``D_embed`` is the embedding dimension.
        Returns:
            Token-level CNN features of shape ``(B, L, output_dim)``.
        """
        ### apply each convolution over the token sequence ###
        # dim(conv_out) = (B, L, filters_i)
        conv_outputs = [conv(token_embeddings) for conv in self.convs]

        ### concatenate all the convolution outputs ###
        # dim(features) = (B, L, output_dim)
        features = tf.concat(conv_outputs, axis=-1)

        return features


class Branch3(tf.keras.layers.Layer):
    """
    Joint encoder branch for the RNN-based model.

    Branch3 embeds a joint token sequence, extracts local token-level n-gram
    features with a TokenCNN, concatenates the original token embeddings with the
    CNN features, applies dropout, and encodes the combined sequence with a
    bidirectional LSTM.

    This branch is intended to model the joint question-context sequence while
    preserving one contextual representation per token position.

    Padding tokens with ID ``0`` are masked during recurrent encoding so they do
    not affect the sequence representations.

    Input shape:
        ``(B, L)``

    Output shape:
        ``(B, L, 2 * output_dim)``
    """

    def __init__(self, embedder: Embedder, **kwargs: object) -> None:
        """
        Initialize the Branch3 joint encoder.

        Loads the branch-specific hyperparameters from the model-definition config,
        stores the shared token embedder, and constructs the token-level CNN,
        dropout layer, and bidirectional LSTM encoder.

        Args:
            embedder: Shared token embedding layer used to map token IDs to dense
                vectors.
            **kwargs: Additional keyword arguments passed to
                ``tf.keras.layers.Layer``.

        Returns:
            None.
        """
        ### init ###
        super().__init__(**kwargs)

        ### load up the config ###
        config = load_pipeline_config(PipelineStep.model_definition)

        ### get all the config values ###
        dropout: float = get_config_value(config, "rnn_branch_3", "dropout", 0.2)
        output_dim: int = get_config_value(config, "rnn_branch_3", "output_dim", 128)

        ### define the layers ###
        self.embedder = embedder
        self.token_cnn = TokenCNN(name="branch3_token_cnn")
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(output_dim, return_sequences=True),
            name="branch3_lstm_encoder",
        )

    def call(
        self, token_ids: tf.Tensor, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Encode a batch of joint token sequences.

        Creates a padding mask from ``token_ids != 0``, embeds the token IDs,
        extracts token-level CNN features, concatenates the embeddings and CNN
        features, applies dropout, and encodes the resulting sequence with a
        bidirectional LSTM.

        Args:
            token_ids: Integer token ID tensor of shape ``(B, L)``, where ``B`` is
                the batch size and ``L`` is the sequence length. Token ID ``0`` is
                treated as padding.
            training: Whether the layer is running in training mode. Controls
                dropout behavior.

        Returns:
            A tuple containing:

            - The encoded sequence tensor of shape ``(B, L, 2 * output_dim)``.
            - The padding mask tensor of shape ``(B, L)``.
        """
        ### define the mask ###
        # 0 here is for padding tokens
        mask: tf.Tensor = tf.not_equal(token_ids, 0)

        ### get the token embeddings ###
        # dim(tok) = (B, L, D_embed)
        tok = self.embedder(token_ids)

        ### extract local token-level features using the token CNN ###
        # dim(cnn_features) = (B, L, D_token_cnn)
        cnn_features = self.token_cnn(tok)

        ### concatenate token embeddings with token CNN features ###
        # dim(x) = (B, L, D_embed + D_token_cnn)
        x = tf.concat([tok, cnn_features], axis=-1)

        ### apply dropout to the combined features ###
        x = self.dropout(x, training=training)

        ### encode the sequence with the bidirectional LSTM ###
        # dim(seq) = (B, L, 2*D_out) because it's bidirectional
        seq = self.encoder(x, mask=mask, training=training)

        return seq, mask


if __name__ == "__main__":
    pass
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/
