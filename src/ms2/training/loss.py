from __future__ import annotations

import math
from collections.abc import Sequence


def _softmax(values: Sequence[float]) -> list[float]:
    shifted = [float(value) for value in values]
    max_value = max(shifted)
    exp_values = [math.exp(value - max_value) for value in shifted]
    denom = sum(exp_values)
    return [value / denom for value in exp_values]


def label_smoothed_cross_entropy(
    logits: Sequence[Sequence[float]],
    targets: Sequence[int],
    mask: Sequence[bool],
    smoothing: float = 0.1,
) -> float | object:
    """Masked label-smoothed cross entropy reduced over unmasked tokens.

    Uses TensorFlow ops when available for differentiable training graphs,
    with a deterministic Python fallback for environments without TensorFlow.
    """
    if not 0 <= smoothing < 1:
        raise ValueError("smoothing must be in [0, 1)")
    if len(logits) != len(targets) or len(targets) != len(mask):
        raise ValueError("logits, targets, and mask must have equal length")
    if not logits:
        return 0.0

    try:
        import tensorflow as tf  # type: ignore[import-not-found]
    except ImportError:
        tf = None  # type: ignore[assignment]

    if tf is not None:
        logits_tensor = tf.convert_to_tensor(logits, dtype=tf.float32)
        targets_tensor = tf.convert_to_tensor(targets, dtype=tf.int32)
        mask_tensor = tf.cast(tf.convert_to_tensor(mask), tf.float32)

        vocab_size = tf.shape(logits_tensor)[-1]
        one_hot = tf.one_hot(targets_tensor, depth=vocab_size, dtype=tf.float32)
        off_value = smoothing / tf.cast(tf.maximum(vocab_size - 1, 1), tf.float32)
        smoothed = one_hot * (1.0 - smoothing) + (1.0 - one_hot) * off_value

        token_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=smoothed,
            logits=logits_tensor,
        )
        masked_loss = token_loss * mask_tensor
        token_count = tf.reduce_sum(mask_tensor)
        return tf.where(
            token_count > 0,
            tf.reduce_sum(masked_loss) / token_count,
            tf.constant(0.0, dtype=tf.float32),
        )

    total_loss = 0.0
    token_count = 0
    for row, target, keep in zip(logits, targets, mask, strict=True):
        if not keep:
            continue
        vocab_size = len(row)
        if vocab_size <= 1:
            raise ValueError("each logits row must have at least two classes")
        if target < 0 or target >= vocab_size:
            raise ValueError("target index out of range")

        probabilities = _softmax(row)
        off_value = smoothing / float(vocab_size - 1)
        row_loss = 0.0
        for class_index, prob in enumerate(probabilities):
            target_prob = 1.0 - smoothing if class_index == target else off_value
            row_loss -= target_prob * math.log(max(prob, 1e-12))
        total_loss += row_loss
        token_count += 1

    if token_count == 0:
        return 0.0
    return float(total_loss / token_count)
