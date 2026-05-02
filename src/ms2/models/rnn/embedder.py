### ~~~ GLOBAL IMPORT ~~~ ###
from pathlib import Path
import tensorflow as tf
import json

### ~~~ LOCAL IMPORT ~~~ ###
from src.ms2.util import (
    data_path,
    DataSteps,
    PipelineStep,
    load_pipeline_config,
    get_config_value,
)

### ~~~ STATE MANAGEMENT ~~~ ###
# None


def get_vocab_size() -> int:
    """"""
    ### get the path using the data_path ###
    root_path: Path = data_path[DataSteps.interim]
    vocab_path: Path = root_path / "vocab.json"

    ### load up the vocab ###
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    ### get the vocab size using the token_to_id mapping ###
    vocab_size: int = len(vocab["token_to_id"])

    return vocab_size


class Embedder(tf.keras.layers.Layer):
    def __init__(self) -> None:
        """
        The Embedder class is responsible for creating the token embedding layer. It
        takes in token ids and outputs the corresponding token embeddings. The embedding
        layer is initialized with a random normal distribution with a standard deviation
        of token_embedding_dim^-0.5, which is a common initialization strategy for
        embedding layers. As the model is trained, the embedding layer will learn to
        map token ids to meaningful vector representations that capture the semantic
        relationships between tokens.
        Args:
            None
        Returns:
            None
        Note:
            The embedding layer initialization affects convergence and generalization.
            Initializing embeddings from a normal distribution with standard deviation
            token_embedding_dim^-0.5 keeps their scale stable across embedding sizes,
            reducing the risk of very large or very small activations and helping avoid
            vanishing or exploding gradients during training.
        """
        ### init ###
        super().__init__()

        ### load up the config ###
        config = load_pipeline_config(PipelineStep.model_definition)

        ### get the vocab size and token embedding dim ###
        vocab_size: int = get_vocab_size()
        token_embedding_dim: int = get_config_value(
            config, "embedder", "token_embedding_dim", 128
        )

        ### create the token embedding layer ###
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=token_embedding_dim,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                stddev=token_embedding_dim**-0.5
            ),
            mask_zero=True,
            name="token_embedding",
        )

    @property
    def embeddings(self) -> tf.Variable:
        return self.embedding.embeddings

    def call(self, token_ids: tf.Tensor) -> tf.Tensor:
        return self.embedding(token_ids)


if __name__ == "__main__":
    pass
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/
