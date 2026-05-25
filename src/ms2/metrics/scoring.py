from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from src.ms2.metrics.normalize import arabic_post_normalize


def exact_match(prediction: str, reference: str) -> float:
    """Return normalized exact-match for one prediction/reference pair."""
    return float(arabic_post_normalize(prediction) == arabic_post_normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Return SQuAD-style token F1 after ADR §1.8 normalization."""
    pred_tokens = arabic_post_normalize(prediction).split()
    ref_tokens = arabic_post_normalize(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def char_edit_distance_normalized(prediction: str, reference: str) -> float:
    """Return Levenshtein distance divided by max normalized string length."""
    pred = arabic_post_normalize(prediction)
    ref = arabic_post_normalize(reference)
    denom = max(len(pred), len(ref))
    if denom == 0:
        return 0.0
    return _levenshtein(pred, ref) / denom


def bleu1(prediction: str, reference: str) -> float:
    """Return unigram BLEU with brevity penalty after normalization."""
    pred_tokens = arabic_post_normalize(prediction).split()
    ref_tokens = arabic_post_normalize(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    clipped = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    precision = clipped / len(pred_tokens)
    if precision == 0.0:
        return 0.0
    brevity = (
        1.0
        if len(pred_tokens) > len(ref_tokens)
        else pow(2.718281828459045, 1 - len(ref_tokens) / len(pred_tokens))
    )
    return brevity * precision


def aggregate_metrics(
    predictions: Sequence[str], references: Sequence[str]
) -> dict[str, float]:
    """Aggregate MS2 metric values into a RunSummary-compatible metric dict."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return {"em": 0.0, "token_f1": 0.0, "char_edit_distance": 0.0, "bleu1": 0.0}
    pairs = list(zip(predictions, references, strict=True))
    return {
        "em": sum(exact_match(pred, ref) for pred, ref in pairs) / len(pairs),
        "token_f1": sum(token_f1(pred, ref) for pred, ref in pairs) / len(pairs),
        "char_edit_distance": sum(
            char_edit_distance_normalized(pred, ref) for pred, ref in pairs
        )
        / len(pairs),
        "bleu1": sum(bleu1(pred, ref) for pred, ref in pairs) / len(pairs),
    }


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + int(left_char != right_char),
                )
            )
        previous = current
    return previous[-1]
