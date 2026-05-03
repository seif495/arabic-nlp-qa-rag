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


class LearnedPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs: object) -> None:
        """
        The LearnedPooling class implements a learned pooling mechanism that computes a
        weighted average of the input sequence. It uses a dense layer to compute scores
        for each time step in the input sequence, applies a softmax to get
        probabilities, and then computes a weighted sum of the input sequence based on
        these probabilities. The pooling activation function can be configured through
        the pipeline configuration, allowing for flexibility in how the scores are
        computed.
        Args:
            **kwargs: Additional keyword arguments for the layer.
        Returns:
            None
        """
        ### init ###
        super().__init__(**kwargs)

        ### load up the config ###
        config = load_pipeline_config(PipelineStep.model_definition)

        ### get all the config values ###
        pooling_activation: str = get_config_value(
            config, "rnn_branch-1", "pooling_activation", "tanh"
        )

        ### define the temporal projection layer ###
        self.score = tf.keras.layers.Dense(
            1,
            activation=pooling_activation,
            use_bias=False,
        )

    def call(self, sequence: tf.Tensor, mask: tf.Tensor | None = None) -> tf.Tensor:
        """
        Pool a variable-length sequence into a single fixed-size vector.
        Computes one learned scalar score for each timestep, optionally masks out
        padding positions, normalizes the scores with a softmax over the sequence
        length, and returns the weighted sum of the input vectors.
        Args:
            sequence: Input tensor of shape ``(B, L, D)``, where ``B`` is the batch
                size, ``L`` is the sequence length, and ``D`` is the feature size.
            mask: Optional boolean or numeric mask of shape ``(B, L)``. Positions
                evaluating to ``False`` are treated as padding and receive zero
                probability after softmax.
        Returns:
            A pooled tensor of shape ``(B, D)`` containing one weighted sequence
            representation per batch item.
        """
        ### compute the scores for each time step in the sequence ###
        # dim(input) = (B, L, D)
        # dim(score) = (B, L)
        scores = tf.squeeze(self.score(sequence), axis=-1)

        ### apply the mask so padding tokens do not contribute to the scores ###
        if mask is not None:
            scores = tf.where(
                tf.cast(mask, dtype=tf.bool),
                scores,
                tf.fill(
                    tf.shape(scores),
                    tf.constant(-1e9, dtype=scores.dtype),
                ),
            )

        ### compute the softmax over the scores to get probs over time ###
        # dim(probs) = (B, L)
        probs = tf.nn.softmax(tf.cast(scores, dtype=tf.float32), axis=-1)
        probs = tf.cast(probs, dtype=sequence.dtype)

        ### finally reduce the dim on the length ###
        # dim(pooled) = (B, D)
        pooled = tf.reduce_sum(sequence * tf.expand_dims(probs, axis=-1), axis=1)

        return pooled


class Branch1(tf.keras.layers.Layer):
    def __init__(self, embedder: Embedder, **kwargs: object) -> None:
        """"""
        ### init ###
        super().__init__(**kwargs)

        ### load up the config ###
        config = load_pipeline_config(PipelineStep.model_definition)

        ### get all the config values ###
        dropout: float = get_config_value(config, "rnn_branch-1", "dropout", 0.2)
        output_dim: int = get_config_value(config, "rnn_branch-1", "output_dim", 128)

        ### define the layers ###
        self.embedder = embedder
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(output_dim, return_sequences=True),
            name="branch1_lstm_encoder",
        )
        self.pool = LearnedPooling(name="branch1_structured_pool")

    def call(self, token_ids: tf.Tensor, training: bool = False):
        """"""
        ### define the mask ###
        # 0 here is for padding tokens
        mask: tf.Tensor = tf.not_equal(token_ids, 0)

        ### get the token embeddings ###
        # dim(x) = (B, L, D_embed)
        x = self.dropout(self.embedder(token_ids), training=training)

        ### encode the sequence with the bidirectional LSTM ###
        # dim(seq) = (B, L, 2*D_out) because it's bidirectional
        seq = self.encoder(x, mask=mask, training=training)

        ### pool the sequence into a single vector ###
        # dim(pooled) = (B, 2*D_out)
        pooled = self.pool(seq, mask=mask)

        return pooled


if __name__ == "__main__":
    pass
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/
