### ~~~ GLOBAL IMPORT ~~~ ###
import tensorflow as tf

### ~~~ LOCAL IMPORT ~~~ ###
from src.ms2.models.rnn.embedder import Embedder
from src.ms2.models.rnn.encoder_stack import Encoder
from src.ms2.models.rnn.decoder import FiLMGenerator, Decoder

### ~~~ STATE MANAGEMENT ~~~ ###
# None


class DecoderStack(tf.keras.layers.Layer):
    """
    End-to-end encoder-conditioned decoder stack for teacher-forced training.

    This layer wires:

    - tri-branch gated encoder (produces sequence + summary)
    - FiLM generator (produces ``gamma`` and ``beta``)
    - two-layer LSTM decoder (produces token logits)

    Input shapes:
        ``question_ids``: ``(B, L_q)``
        ``context_ids``: ``(B, L_c)``
        ``joint_ids``: ``(B, L_joint)``
        ``decoder_inputs``: ``(B, L_dec)``

    Output shapes:
        ``logits``: ``(B, L_dec, V)``
    """

    def __init__(self, embedder: Embedder | None = None, **kwargs: object) -> None:
        """
        Initialize the decoder stack.

        Args:
            embedder: Optional shared embedder across encoder and decoder.
                If omitted, a new ``Embedder`` is created.
            **kwargs: Additional keyword arguments passed to
                ``tf.keras.layers.Layer``.

        Returns:
            None.
        """
        ### init ###
        super().__init__(**kwargs)

        ### create one shared token embedder ###
        self.embedder = embedder if embedder is not None else Embedder()

        ### construct encoder + conditioning + decoder blocks ###
        self.encoder = Encoder(embedder=self.embedder, name="encoder")
        self.film_generator = FiLMGenerator(name="film_generator")
        self.decoder = Decoder(embedder=self.embedder, name="decoder")

    def call(
        self,
        question_ids: tf.Tensor,
        context_ids: tf.Tensor,
        joint_ids: tf.Tensor,
        decoder_inputs: tf.Tensor,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """
        Run the full decoder stack and return logits with key intermediates.

        Args:
            question_ids: Question token IDs of shape ``(B, L_q)``.
            context_ids: Context token IDs of shape ``(B, L_c)``.
            joint_ids: Joint token IDs of shape ``(B, L_joint)``.
            decoder_inputs: Teacher-forced decoder input IDs of shape
                ``(B, L_dec)``.
            training: Whether the stack is running in training mode.

        Returns:
            Dictionary containing:

            - ``logits``: ``(B, L_dec, V)``
            - ``encoder_output``: ``(B, L_c, D_fuse)``
            - ``encoder_summary``: ``(B, D_fuse)``
            - ``encoder_mask``: ``(B, L_c)``
            - ``gates``: ``(B, L_c, 3)``
            - ``gamma``: ``(B, D_dec)``
            - ``beta``: ``(B, D_dec)``
        """
        ### encode question/context/joint sequence ###
        encoder_outputs = self.encoder(
            question_ids=question_ids,
            context_ids=context_ids,
            joint_ids=joint_ids,
            training=training,
        )

        ### get global encoder summary for conditioning ###
        encoder_summary = encoder_outputs["encoder_summary"]

        ### generate FiLM parameters from encoder summary ###
        gamma, beta = self.film_generator(encoder_summary)

        ### decode answer-prefix tokens into logits ###
        logits = self.decoder(
            decoder_inputs=decoder_inputs,
            encoder_summary=encoder_summary,
            gamma=gamma,
            beta=beta,
            training=training,
        )

        return {
            "logits": logits,
            "encoder_output": encoder_outputs["encoder_output"],
            "encoder_summary": encoder_summary,
            "encoder_mask": encoder_outputs["encoder_mask"],
            "gates": encoder_outputs["gates"],
            "gamma": gamma,
            "beta": beta,
        }


def main() -> None:
    """
    Minimal dry-run for the decoder stack.

    This validates that all major tensor dimensions line up across:
    encoder -> summary -> FiLM -> decoder -> logits.
    """
    ### define synthetic dimensions ###
    vocab_size = 128
    batch_size = 2
    l_q = 8
    l_c = 12
    l_joint = l_q + l_c + 3
    l_dec = 7

    ### build stack and Keras wrapper model ###
    shared_embedder = Embedder(vocab_size=vocab_size)
    stack = DecoderStack(embedder=shared_embedder, name="decoder_stack")

    question_in = tf.keras.Input(shape=(l_q,), dtype=tf.int32, name="question_ids")
    context_in = tf.keras.Input(shape=(l_c,), dtype=tf.int32, name="context_ids")
    joint_in = tf.keras.Input(shape=(l_joint,), dtype=tf.int32, name="joint_ids")
    decoder_in = tf.keras.Input(shape=(l_dec,), dtype=tf.int32, name="decoder_inputs")

    outputs = stack(
        question_ids=question_in,
        context_ids=context_in,
        joint_ids=joint_in,
        decoder_inputs=decoder_in,
    )

    model = tf.keras.Model(
        inputs=[question_in, context_in, joint_in, decoder_in],
        outputs=outputs,
        name="rnn_decoder_stack_smoke",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam())

    ### create random non-padding ids ###
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

    ### run one forward pass and print shapes ###
    out = model([question_ids, context_ids, joint_ids, decoder_inputs], training=False)
    print(f"logits shape: {out['logits'].shape}")
    print(f"encoder_output shape: {out['encoder_output'].shape}")
    print(f"encoder_summary shape: {out['encoder_summary'].shape}")
    print(f"encoder_mask shape: {out['encoder_mask'].shape}")
    print(f"gates shape: {out['gates'].shape}")
    print(f"gamma shape: {out['gamma'].shape}")
    print(f"beta shape: {out['beta'].shape}")
    print("decoder stack dry-run: ok")


if __name__ == "__main__":
    main()
