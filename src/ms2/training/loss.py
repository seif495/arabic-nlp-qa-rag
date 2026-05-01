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
) -> float:
    """Masked label-smoothed cross entropy reduced over unmasked tokens."""
    if not 0 <= smoothing < 1:
        raise ValueError("smoothing must be in [0, 1)")
    if len(logits) != len(targets) or len(targets) != len(mask):
        raise ValueError("logits, targets, and mask must have equal length")
    if not logits:
        return 0.0

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
