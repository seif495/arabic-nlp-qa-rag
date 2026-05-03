### ~~~ GLOBAL IMPORT ~~~ ###
import tensorflow as tf

### ~~~ LOCAL IMPORT ~~~ ###
from src.ms2.models.rnn.embedder import Embedder
from src.ms2.models.rnn.encoder import (
    Branch1,
    Branch2,
    Branch3,
    BranchAligner,
    GatedMerge,
)

### ~~~ STATE MANAGEMENT ~~~ ###
# None


class Encoder(tf.keras.layers.Layer):
    """
    Full tri-branch encoder stack for Model A.

    This layer composes the current encoder-side architecture defined in the ADR:

    - Branch 1: question encoder (pooled question summary)
    - Branch 2: context encoder (context sequence)
    - Branch 3: joint encoder over ``[<bos>] q [<sep>] c [<eos>]``
    - BranchAligner: aligns all branch outputs on the context axis
    - GatedMerge: per-position softmax fusion over the three branches

    Inputs:
        question_ids: ``(B, L_q)``
        context_ids: ``(B, L_c)``
        joint_ids: ``(B, L_joint)``

    Outputs:
        fused: ``(B, L_c, D_fuse)``
        gates: ``(B, L_c, 3)``
    """

    def __init__(self, embedder: Embedder | None = None, **kwargs: object) -> None:
        """
        Initialize the composed encoder stack.

        Args:
            embedder: Optional shared token embedder. If omitted, a new Embedder is
                created and shared across all three branches.
            **kwargs: Additional keyword arguments passed to
                ``tf.keras.layers.Layer``.

        Returns:
            None.
        """
        super().__init__(**kwargs)

        ### define a shared token embedding layer ###
        # All branches use the same token embedding space so the branch outputs are
        # semantically comparable before alignment and gated fusion.
        self.embedder = embedder if embedder is not None else Embedder()

        ### construct encoder branches ###
        self.branch_1 = Branch1(embedder=self.embedder, name="branch_1")
        self.branch_2 = Branch2(embedder=self.embedder, name="branch_2")
        self.branch_3 = Branch3(embedder=self.embedder, name="branch_3")

        ### construct post-branch composition blocks ###
        self.aligner = BranchAligner(name="branch_aligner")
        self.gated_merge = GatedMerge(name="gated_merge")

    def call(
        self,
        question_ids: tf.Tensor,
        context_ids: tf.Tensor,
        joint_ids: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Run the encoder stack end-to-end.

        This method computes branch-specific encodings, aligns all branches onto the
        context-token axis, and returns the gated fused sequence.

        Args:
            question_ids: Question token IDs of shape ``(B, L_q)``.
            context_ids: Context token IDs of shape ``(B, L_c)``.
            joint_ids: Joint-sequence token IDs of shape ``(B, L_joint)`` using the
                layout ``[<bos>] question [<sep>] context [<eos>]``.
            training: Whether the layer is running in training mode. Controls dropout.

        Returns:
            A tuple ``(fused, gates)`` where:
            - ``fused`` has shape ``(B, L_c, D_fuse)``.
            - ``gates`` has shape ``(B, L_c, 3)``.
        """
        ### compute true question lengths (excluding padding) ###
        # BranchAligner needs this to locate where context starts inside the joint
        # sequence representation produced by Branch 3.
        question_lengths = tf.reduce_sum(
            tf.cast(tf.not_equal(question_ids, 0), dtype=tf.int32), axis=1
        )

        ### run all three branches ###
        # dim(b1_pooled) = (B, D1)
        # dim(b2_seq)    = (B, L_c, D2)
        # dim(b3_seq)    = (B, L_joint, D3)
        b1_pooled = self.branch_1(question_ids, training=training)
        b2_seq = self.branch_2(context_ids, training=training)
        b3_seq, _ = self.branch_3(joint_ids, training=training)

        ### align branch outputs to context axis ###
        # After alignment:
        #   b1_aligned: (B, L_c, D1)
        #   b2_aligned: (B, L_c, D2)
        #   b3_aligned: (B, L_c, D3)
        aligned = self.aligner(
            b1_pooled=b1_pooled,
            b2_seq=b2_seq,
            b3_seq=b3_seq,
            question_lengths=question_lengths,
        )

        ### fuse branches using per-position gated merge ###
        # dim(fused) = (B, L_c, D_fuse)
        # dim(gates) = (B, L_c, 3)
        fused, gates = self.gated_merge(aligned, training=training)
        return fused, gates


def main() -> None:
    """
    Minimal smoke test for Encoder composition.

    The goal is fast sanity-checking that the composed encoder can:
    - build as a Keras graph,
    - compile,
    - run one synthetic forward pass.
    """
    ### define tiny synthetic dimensions for a quick dry run ###
    vocab_size = 128
    batch_size = 2
    l_q = 8
    l_c = 12
    l_joint = l_q + l_c + 3

    ### build encoder and Keras wrapper model ###
    shared_embedder = Embedder(vocab_size=vocab_size)
    encoder = Encoder(embedder=shared_embedder, name="encoder")

    question_in = tf.keras.Input(shape=(l_q,), dtype=tf.int32, name="question_ids")
    context_in = tf.keras.Input(shape=(l_c,), dtype=tf.int32, name="context_ids")
    joint_in = tf.keras.Input(shape=(l_joint,), dtype=tf.int32, name="joint_ids")

    fused_out, gates_out = encoder(question_in, context_in, joint_in)
    model = tf.keras.Model(
        inputs=[question_in, context_in, joint_in],
        outputs=[fused_out, gates_out],
        name="rnn_encoder_smoke",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam())

    ### create random non-padding token IDs ###
    # minval=1 ensures we avoid ``0`` so this smoke test does not depend on mask
    # edge cases; the goal here is pure shape/graph sanity.
    question_ids = tf.random.uniform(
        shape=(batch_size, l_q), minval=1, maxval=vocab_size, dtype=tf.int32
    )
    context_ids = tf.random.uniform(
        shape=(batch_size, l_c), minval=1, maxval=vocab_size, dtype=tf.int32
    )
    joint_ids = tf.random.uniform(
        shape=(batch_size, l_joint), minval=1, maxval=vocab_size, dtype=tf.int32
    )

    ### run a single forward pass and print key shapes ###
    fused, gates = model([question_ids, context_ids, joint_ids], training=False)
    print(f"fused shape: {fused.shape}")
    print(f"gates shape: {gates.shape}")
    print("encoder dry-run: ok")


if __name__ == "__main__":
    main()
