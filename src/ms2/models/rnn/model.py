### ~~~ GLOBAL IMPORT ~~~ ###
import tensorflow as tf

### ~~~ LOCAL IMPORT ~~~ ###
from src.ms2.models.rnn.embedder import Embedder
from src.ms2.models.rnn.decoder_stack import DecoderStack

### ~~~ STATE MANAGEMENT ~~~ ###
# None


class RNNModel(tf.keras.Model):
    """
    Full RNN-based QA model (Model A) for teacher-forced training.

    This model wraps the end-to-end decoder stack and exposes a clean training
    interface that maps:

    - question/context/joint encoder inputs
    - teacher-forced decoder prefix tokens

    to vocabulary logits over the next-token targets.

    Input shapes:
        ``question_ids``: ``(B, L_q)``
        ``context_ids``: ``(B, L_c)``
        ``joint_ids``: ``(B, L_joint)``
        ``decoder_inputs``: ``(B, L_dec)``

    Output shape:
        ``logits``: ``(B, L_dec, V)``
    """

    def __init__(self, embedder: Embedder | None = None, **kwargs: object) -> None:
        """
        Initialize the full RNN model.

        Args:
            embedder: Optional shared token embedder. If omitted, a new embedder is
                created and shared by encoder and decoder.
            **kwargs: Additional keyword arguments passed to
                ``tf.keras.Model``.

        Returns:
            None.
        """
        ### init ###
        super().__init__(**kwargs)

        ### define one shared embedder and full decoder stack ###
        self.embedder = embedder if embedder is not None else Embedder()
        self.stack = DecoderStack(embedder=self.embedder, name="decoder_stack")

    def call(
        self,
        question_ids: tf.Tensor,
        context_ids: tf.Tensor,
        joint_ids: tf.Tensor,
        decoder_inputs: tf.Tensor,
        training: bool = False,
        return_details: bool = False,
    ) -> tf.Tensor | dict[str, tf.Tensor]:
        """
        Run the full model forward pass.

        Args:
            question_ids: Question token IDs of shape ``(B, L_q)``.
            context_ids: Context token IDs of shape ``(B, L_c)``.
            joint_ids: Joint token IDs of shape ``(B, L_joint)``.
            decoder_inputs: Teacher-forced decoder input IDs of shape
                ``(B, L_dec)``.
            training: Whether the model is in training mode.
            return_details: If ``True``, return a dictionary with logits and key
                intermediate tensors. If ``False``, return logits only.

        Returns:
            Either:
            - ``logits`` of shape ``(B, L_dec, V)`` when ``return_details=False``.
            - A details dictionary when ``return_details=True``.
        """
        ### run full encoder-conditioning-decoder stack ###
        outputs = self.stack(
            question_ids=question_ids,
            context_ids=context_ids,
            joint_ids=joint_ids,
            decoder_inputs=decoder_inputs,
            training=training,
        )

        if return_details:
            return outputs

        return outputs["logits"]


def main() -> None:
    """
    Minimal dry-run for the full RNN model.

    This validates that the top-level model compiles and produces logits with the
    expected shape, while optionally exposing internal tensors for debugging.
    """
    ### define synthetic dimensions ###
    vocab_size = 128
    batch_size = 2
    l_q = 8
    l_c = 12
    l_joint = l_q + l_c + 3
    l_dec = 7

    ### build model and Keras wrapper graph ###
    shared_embedder = Embedder(vocab_size=vocab_size)
    model = RNNModel(embedder=shared_embedder, name="rnn_model")

    question_in = tf.keras.Input(shape=(l_q,), dtype=tf.int32, name="question_ids")
    context_in = tf.keras.Input(shape=(l_c,), dtype=tf.int32, name="context_ids")
    joint_in = tf.keras.Input(shape=(l_joint,), dtype=tf.int32, name="joint_ids")
    decoder_in = tf.keras.Input(shape=(l_dec,), dtype=tf.int32, name="decoder_inputs")

    logits = model(
        question_ids=question_in,
        context_ids=context_in,
        joint_ids=joint_in,
        decoder_inputs=decoder_in,
    )
    graph_model = tf.keras.Model(
        inputs=[question_in, context_in, joint_in, decoder_in],
        outputs=logits,
        name="rnn_model_smoke",
    )
    graph_model.compile(optimizer=tf.keras.optimizers.Adam())

    ### create random non-padding IDs ###
    question_ids = tf.random.uniform(
        shape=(batch_size, l_q), minval=1, maxval=vocab_size, dtype=tf.int32
    )
    context_ids = tf.random.uniform(
        shape=(batch_size, l_c), minval=1, maxval=vocab_size, dtype=tf.int32
    )
    joint_ids = tf.random.uniform(
        shape=(batch_size, l_joint), minval=1, maxval=vocab_size, dtype=tf.int32
    )
    decoder_inputs = tf.random.uniform(
        shape=(batch_size, l_dec), minval=1, maxval=vocab_size, dtype=tf.int32
    )

    ### run one forward pass and print key shape ###
    out = graph_model([question_ids, context_ids, joint_ids, decoder_inputs], training=False)
    print(f"logits shape: {out.shape}")
    print("full model dry-run: ok")


if __name__ == "__main__":
    main()
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/
