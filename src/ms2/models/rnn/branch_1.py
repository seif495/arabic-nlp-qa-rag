### ~~~ GLOBAL IMPORT ~~~ ###
import tensorflow as tf

### ~~~ LOCAL IMPORT ~~~ ###
from src.ms2.util import (
    load_pipeline_config,
    PipelineStep,
    get_config_value,
)

### ~~~ STATE MANAGEMENT ~~~ ###
# None


class LearnedPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs: object) -> None:
        """"""
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
        """"""
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


if __name__ == "__main__":
    pass
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/
