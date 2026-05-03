### ~~~ GLOBAL IMPORT ~~~ ###
import tensorflow as tf

### ~~~ LOCAL IMPORT ~~~ ###
# None

### ~~~ STATE MANAGEMENT ~~~ ###
# None


class BranchAligner(tf.keras.layers.Layer):
    """
    Align the three RNN encoder branches onto the context-token axis.
    The three branches produce representations over different sequence axes:
    - Branch 1 encodes only the question and returns one pooled vector:
        ``b1_pooled: (B, 2 * D_b1)``
    - Branch 2 encodes only the context and already lives on the context axis:
        ``b2_seq: (B, L_c, 2 * D_b2)``
    - Branch 3 encodes the joint sequence:
        ``[<bos>] question [<sep>] context [<eos>]``
        and returns:
        ``b3_seq: (B, L_joint, 2 * D_b3)``
    Before the gated merge block, all three branches must have the same sequence
    length ``L_c`` so that each context position has one representation from each
    branch.

    This layer performs that alignment:

    - Branch 1 is tiled across all context positions.
    - Branch 2 is returned unchanged.
    - Branch 3 is sliced dynamically to keep only the context-token positions.

    Input shapes:
        ``b1_pooled``: ``(B, D1)``
        ``b2_seq``: ``(B, L_c, D2)``
        ``b3_seq``: ``(B, L_joint, D3)``
        ``question_lengths``: ``(B,)``

    Output shapes:
        ``b1_aligned``: ``(B, L_c, D1)``
        ``b2_aligned``: ``(B, L_c, D2)``
        ``b3_aligned``: ``(B, L_c, D3)``
    """

    def __init__(self, **kwargs: object) -> None:
        """
        Initialize the branch alignment layer.
        This layer does not define trainable weights. It only performs tensor
        reshaping, tiling, and dynamic gathering so that all branch outputs are
        aligned to the context axis.
        Args:
            **kwargs: Additional keyword arguments passed to
                ``tf.keras.layers.Layer``.
        Returns:
            None.
        """
        ### init ###
        super().__init__(**kwargs)

    def call(
        self,
        b1_pooled: tf.Tensor,
        b2_seq: tf.Tensor,
        b3_seq: tf.Tensor,
        question_lengths: tf.Tensor,
        context_length: int | tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Align Branch 1, Branch 2, and Branch 3 outputs to the context axis.
        Branch 1 produces a single global question vector, so it is repeated
        across all context positions. Branch 2 already has one vector per context
        token, so it is used as-is. Branch 3 is produced over the full joint
        sequence, so this method gathers only the positions corresponding to the
        context tokens.
        The expected joint sequence layout for Branch 3 is:
            ``[<bos>] question_tokens [<sep>] context_tokens [<eos>]``
        Therefore, for each example, context tokens begin at:
            ``start = question_length + 2``
        The ``+2`` accounts for:
        - one ``<bos>`` token before the question
        - one ``<sep>`` token after the question
        Args:
            b1_pooled: Pooled Branch 1 question representation of shape
                ``(B, D1)``.
            b2_seq: Branch 2 context sequence representation of shape
                ``(B, L_c, D2)``.
            b3_seq: Branch 3 joint sequence representation of shape
                ``(B, L_joint, D3)``.
            question_lengths: Number of non-padding question tokens per example,
                excluding special tokens. Shape ``(B,)``.
            context_length: Optional context length to align to. If omitted, the
                context length is inferred from ``b2_seq.shape[1]``.
        Returns:
            A tuple ``(b1_aligned, b2_aligned, b3_aligned)`` where all three
            tensors are aligned to the context axis and have sequence length
            ``L_c``.
        """
        ### get the context length ###
        # If context_length is not provided, we infer it from Branch 2.
        # Branch 2 is the context-only encoder, so its sequence length is L_c.
        #
        # dim(b2_seq) = (B, L_c, D2)
        # l_c         = L_c
        l_c = (
            tf.shape(b2_seq)[1]
            if context_length is None
            else tf.cast(context_length, dtype=tf.int32)
        )

        ### align branch 1 to the context axis ###
        # Branch 1 gives us one vector for the whole question.
        #
        # dim(b1_pooled)      = (B, D1)
        # dim(b1_pooled[:,None,:]) = (B, 1, D1)
        #
        # We tile this vector L_c times so every context position gets the same
        # question summary.
        #
        # dim(b1_aligned) = (B, L_c, D1)
        b1_aligned = tf.tile(
            b1_pooled[:, None, :],
            multiples=(1, l_c, 1),
        )

        ### branch 2 is already aligned ###
        # Branch 2 is the context encoder, so it already has one vector per
        # context token.
        #
        # dim(b2_aligned) = (B, L_c, D2)
        b2_aligned = b2_seq

        ### compute where the context starts inside the joint branch 3 sequence ###
        # Branch 3 receives the joint sequence:
        #
        #   [<bos>] question_tokens [<sep>] context_tokens [<eos>]
        #
        # For each example:
        #
        #   question_lengths[i] = number of real question tokens
        #
        # Context starts after:
        #
        #   1 token for <bos>
        #   question_lengths[i] question tokens
        #   1 token for <sep>
        #
        # Therefore:
        #
        #   start[i] = question_lengths[i] + 2
        #
        # dim(starts) = (B,)
        starts = tf.cast(question_lengths, dtype=tf.int32) + 2

        ### build the context offsets for every batch item ###
        # For each example, we need to gather positions:
        #
        #   start[i] + 0
        #   start[i] + 1
        #   start[i] + 2
        #   ...
        #   start[i] + L_c - 1
        #
        # tf.range(l_c) gives:
        #
        #   [0, 1, 2, ..., L_c - 1]
        #
        # Broadcasting starts[:, None] with tf.range(l_c)[None, :] gives:
        #
        # dim(offsets) = (B, L_c)
        offsets = starts[:, None] + tf.range(l_c, dtype=tf.int32)[None, :]

        ### build batch indices for gather_nd ###
        # tf.gather_nd needs indices of the form:
        #
        #   [batch_index, sequence_position]
        #
        # So for each context position, we need to pair the sequence offset with
        # its batch row.
        #
        # Example for B = 2 and L_c = 4:
        #
        #   batch_ids =
        #   [[0, 0, 0, 0],
        #    [1, 1, 1, 1]]
        #
        # dim(batch_ids) = (B, L_c)
        batch_ids = tf.range(tf.shape(b3_seq)[0], dtype=tf.int32)[:, None]
        batch_ids = tf.tile(batch_ids, multiples=(1, l_c))

        ### combine batch ids and sequence offsets ###
        # Each index now points to one context token inside b3_seq.
        #
        # dim(batch_ids)  = (B, L_c)
        # dim(offsets)    = (B, L_c)
        #
        # dim(gather_idx) = (B, L_c, 2)
        #
        # gather_idx[i, j] = [i, starts[i] + j]
        gather_idx = tf.stack([batch_ids, offsets], axis=-1)

        ### gather the context part from branch 3 ###
        # dim(b3_seq)     = (B, L_joint, D3)
        # dim(gather_idx) = (B, L_c, 2)
        #
        # Each gather index selects one vector from b3_seq.
        #
        # dim(b3_aligned) = (B, L_c, D3)
        b3_aligned = tf.gather_nd(b3_seq, gather_idx)

        return b1_aligned, b2_aligned, b3_aligned


if __name__ == "__main__":
    pass
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/
