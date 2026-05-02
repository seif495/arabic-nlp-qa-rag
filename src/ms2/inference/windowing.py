from __future__ import annotations

from collections.abc import Callable, Sequence

from src.ms2.inference.greedy import DecodeResult


def sliding_context_windows(
    tokens: Sequence[int], l_c: int = 384
) -> list[tuple[int, list[int]]]:
    """Split contexts with stride `L_c / 2`, ADR §1.3."""
    if l_c <= 0:
        raise ValueError("l_c must be positive")
    stride = max(1, l_c // 2)
    if len(tokens) <= l_c:
        return [(0, list(tokens))]
    windows: list[tuple[int, list[int]]] = []
    start = 0
    while start < len(tokens):
        window = list(tokens[start : start + l_c])
        windows.append((start, window))
        if start + l_c >= len(tokens):
            break
        start += stride
    return windows


def select_best_window(
    tokens: Sequence[int],
    decode_window: Callable[[list[int]], DecodeResult],
    l_c: int = 384,
) -> tuple[int, DecodeResult]:
    """Decode all windows and select by length-normalized score."""
    scored = [
        (start, decode_window(window))
        for start, window in sliding_context_windows(tokens, l_c=l_c)
    ]
    return max(scored, key=lambda item: item[1].normalized_score)
