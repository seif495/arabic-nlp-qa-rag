### ~~~ GLOBAL IMPORT ~~~ ###
import tensorflow as tf

### ~~~ LOCAL IMPORT ~~~ ###
# None

### ~~~ STATE MANAGEMENT ~~~ ###
# None


def masked_mean(sequence: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """
    Compute the mean of a sequence while ignoring padded positions.

    Args:
        sequence: Tensor of shape ``(B, L, D)``.
        mask: Boolean mask of shape ``(B, L)`` where True means keep and False
            means ignore.

    Returns:
        Tensor of shape ``(B, D)``.
    """
    ### cast mask to the same dtype as the sequence ###
    # dim(mask) = (B, L)
    mask = tf.cast(mask, dtype=sequence.dtype)

    ### add a final singleton dimension so it can multiply sequence ###
    # dim(mask) = (B, L, 1)
    mask = mask[..., None]

    ### zero out padded positions ###
    # dim(masked_sequence) = (B, L, D)
    masked_sequence = sequence * mask

    ### sum only the real token vectors ###
    # dim(numerator) = (B, D)
    numerator = tf.reduce_sum(masked_sequence, axis=1)

    ### count how many real tokens each example has ###
    # dim(denominator) = (B, 1)
    denominator = tf.reduce_sum(mask, axis=1)

    ### avoid division by zero just in case an example is fully padded ###
    denominator = tf.maximum(denominator, tf.constant(1.0, dtype=sequence.dtype))

    ### compute the average over real tokens only ###
    # dim(summary) = (B, D)
    summary = numerator / denominator

    return summary
