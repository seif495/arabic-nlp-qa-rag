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


class GatedMerge(tf.keras.layers.Layer):
    """
    Per-position gated merge block for fusing the three RNN encoder branches.

    GatedMerge receives the three branch outputs after they have already been
    aligned to the context axis. Each branch must therefore have the same batch
    size and sequence length:

    - Branch 1 aligned question summary:
        ``b1_aligned: (B, L_c, D1)``

    - Branch 2 context sequence:
        ``b2_aligned: (B, L_c, D2)``

    - Branch 3 aligned joint-sequence context slice:
        ``b3_aligned: (B, L_c, D3)``

    The layer first projects all three branches into a shared fusion dimension,
    then computes three softmax gates per context position. These gates decide how
    much each branch contributes to the fused representation at that position.

    This is a mixture-of-experts style merge, not attention: the softmax is over
    the three branch identities, not over sequence positions.

    Input shapes:
        ``branches[0]``: ``(B, L_c, D1)``
        ``branches[1]``: ``(B, L_c, D2)``
        ``branches[2]``: ``(B, L_c, D3)``

    Output shapes:
        ``fused``: ``(B, L_c, D_FUSE)``
        ``gates``: ``(B, L_c, 3)``
    """

    def __init__(self, **kwargs: object) -> None:
        """
        Initialize the gated branch merge block.

        Loads the merge hyperparameters from the model-definition config and
        constructs one projection layer per branch, a small gate MLP, dropout, and
        layer normalization.

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
        fusion_dim: int = get_config_value(config, "rnn_gated_merge", "fusion_dim", 192)
        dropout_ff: float = get_config_value(
            config, "rnn_gated_merge", "dropout_ff", 0.3
        )
        mean_merge: bool = get_config_value(
            config, "rnn_gated_merge", "mean_merge", False
        )

        ### store the config values needed during the forward pass ###
        self.fusion_dim = fusion_dim
        self.mean_merge = mean_merge

        ### project every branch into the same fusion dimension ###
        # Before merging, all branches must live in the same feature space.
        #
        # dim(branch_1) = (B, L_c, D1)
        # dim(h1)      = (B, L_c, fusion_dim)
        #
        # dim(branch_2) = (B, L_c, D2)
        # dim(h2)      = (B, L_c, fusion_dim)
        #
        # dim(branch_3) = (B, L_c, D3)
        # dim(h3)      = (B, L_c, fusion_dim)
        self.proj1 = tf.keras.layers.Dense(fusion_dim, name="merge_proj_branch1")
        self.proj2 = tf.keras.layers.Dense(fusion_dim, name="merge_proj_branch2")
        self.proj3 = tf.keras.layers.Dense(fusion_dim, name="merge_proj_branch3")

        ### define the gate MLP hidden layer ###
        # The gate network sees the concatenated projected branch features:
        #
        #   concat([h1, h2, h3], axis=-1)
        #
        # dim(concat) = (B, L_c, 3 * fusion_dim)
        # dim(hidden) = (B, L_c, fusion_dim)
        self.gate_hidden = tf.keras.layers.Dense(
            fusion_dim,
            activation=tf.keras.activations.gelu,
            name="merge_gate_hidden",
        )

        ### define the gate output layer ###
        # This produces three logits per context position:
        #
        #   logit_0 = score for Branch 1
        #   logit_1 = score for Branch 2
        #   logit_2 = score for Branch 3
        #
        # dim(gate_logits) = (B, L_c, 3)
        self.gate_out = tf.keras.layers.Dense(3, name="merge_gate_logits")

        ### define post-merge regularization and normalization ###
        self.dropout = tf.keras.layers.Dropout(dropout_ff)
        self.norm = tf.keras.layers.LayerNormalization(name="merge_layer_norm")

    def call(
        self,
        branches: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Fuse the aligned branch representations with per-position softmax gates.

        Each branch is projected to ``D_FUSE`` dimensions. The projected branch
        vectors are then concatenated and passed through a gate MLP that produces
        three logits at each context position. A softmax turns those logits into
        branch weights, and the final representation is the weighted sum of the
        three projected branch vectors.

        Args:
            branches: Tuple containing the three aligned branch tensors:
                ``(b1_aligned, b2_aligned, b3_aligned)``. Each tensor must have
                shape ``(B, L_c, D_i)`` and the same ``B`` and ``L_c``.
            training: Whether the layer is running in training mode. Controls
                dropout behavior.

        Returns:
            A tuple containing:

            - The fused context representation of shape ``(B, L_c, D_FUSE)``.
            - The branch gate weights of shape ``(B, L_c, 3)``.
        """
        ### project branch 1 into the shared fusion space ###
        # Branch 1 is the tiled question summary.
        #
        # dim(branches[0]) = (B, L_c, D1)
        # dim(h1)          = (B, L_c, D_FUSE)
        h1 = self.proj1(branches[0])

        ### project branch 2 into the shared fusion space ###
        # Branch 2 is the context-only recurrent sequence.
        #
        # dim(branches[1]) = (B, L_c, D2)
        # dim(h2)          = (B, L_c, D_FUSE)
        h2 = self.proj2(branches[1])

        ### project branch 3 into the shared fusion space ###
        # Branch 3 is the joint question-context sequence sliced to context tokens.
        #
        # dim(branches[2]) = (B, L_c, D3)
        # dim(h3)          = (B, L_c, D_FUSE)
        h3 = self.proj3(branches[2])

        ### compute branch gates ###
        # There are two supported modes:
        #
        #   1. learned gated merge:
        #        gates are predicted from [h1; h2; h3]
        #
        #   2. mean merge ablation:
        #        gates are fixed to [1/3, 1/3, 1/3]
        #
        # In both cases:
        #
        #   dim(gates) = (B, L_c, 3)
        if self.mean_merge:
            ### create fixed uniform gates for the ablation ###
            # We need a tensor with the same batch and sequence dimensions as h1,
            # but with a final dimension of 3 for the three branches.
            #
            # dim(tf.shape(h1)[:-1]) = [B, L_c]
            # dim(gate_shape)       = [B, L_c, 3]
            gate_shape = tf.concat(
                [tf.shape(h1)[:-1], tf.constant([3], dtype=tf.int32)],
                axis=0,
            )

            ### assign equal probability to every branch ###
            # Every context position uses:
            #
            #   Branch 1 weight = 1/3
            #   Branch 2 weight = 1/3
            #   Branch 3 weight = 1/3
            gates = tf.ones(gate_shape, dtype=h1.dtype) / tf.cast(3.0, dtype=h1.dtype)

        else:
            ### concatenate projected branch features ###
            # The gate network gets to inspect all three branch vectors at the
            # current context position before deciding the mixture weights.
            #
            # dim(h1)       = (B, L_c, D_FUSE)
            # dim(h2)       = (B, L_c, D_FUSE)
            # dim(h3)       = (B, L_c, D_FUSE)
            #
            # dim(merged_in) = (B, L_c, 3 * D_FUSE)
            merged_in = tf.concat([h1, h2, h3], axis=-1)

            ### compute hidden gate features ###
            # dim(hidden) = (B, L_c, D_FUSE)
            hidden = self.gate_hidden(merged_in)

            ### compute raw branch gate logits ###
            # dim(gate_logits) = (B, L_c, 3)
            gate_logits = self.gate_out(hidden)

            ### normalize logits into branch probabilities ###
            # Softmax is computed in float32 for numerical stability, then cast
            # back to the branch dtype. This is useful under mixed precision.
            #
            # The softmax axis is the branch axis, not the sequence axis.
            gates = tf.nn.softmax(tf.cast(gate_logits, dtype=tf.float32), axis=-1)
            gates = tf.cast(gates, dtype=h1.dtype)

        ### compute the weighted mixture of the three projected branches ###
        # gates[..., 0:1] keeps the final dimension so broadcasting works.
        #
        # dim(gates[..., 0:1]) = (B, L_c, 1)
        # dim(h1)              = (B, L_c, D_FUSE)
        #
        # dim(fused)           = (B, L_c, D_FUSE)
        fused = gates[..., 0:1] * h1 + gates[..., 1:2] * h2 + gates[..., 2:3] * h3  # type: ignore

        ### apply dropout and layer normalization ###
        # Dropout is active only during training.
        # LayerNorm keeps the fused representation numerically stable.
        #
        # dim(output) = (B, L_c, D_FUSE)
        output = self.norm(self.dropout(fused, training=training))

        return output, gates


if __name__ == "__main__":
    pass
#    __   _,_ /_ __,
#  _(_/__(_/_/_)(_/(_
#   _/_
#  (/
