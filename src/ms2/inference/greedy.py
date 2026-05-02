from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any


class DecodableModel(Protocol):
    """Small model-agnostic decoding interface required by MS2-INFER-01."""

    def init_state(self, encoder_inputs: Any) -> Any: ...

    def step(self, state: Any, last_token: int) -> tuple[list[float], Any]: ...


@dataclass(frozen=True)
class DecodeResult:
    token_ids: list[int]
    log_probability: float
    normalized_score: float


def greedy_decode(
    model: DecodableModel,
    encoder_inputs: Any,
    bos_id: int = 2,
    eos_id: int = 3,
    max_length: int = 64,
) -> DecodeResult:
    """Greedy decode until EOS or `max_length`, ADR §2.13 / §3.8."""
    state = model.init_state(encoder_inputs)
    last_token = bos_id
    tokens: list[int] = []
    log_prob = 0.0
    for _ in range(max_length):
        logits, state = model.step(state, last_token)
        next_token = max(range(len(logits)), key=lambda idx: logits[idx])
        log_prob += float(logits[next_token])
        tokens.append(next_token)
        last_token = next_token
        if next_token == eos_id:
            break
    return DecodeResult(tokens, log_prob, _length_normalized(log_prob, len(tokens)))


def _length_normalized(log_probability: float, length: int, alpha: float = 0.6) -> float:
    # Wu et al. length normalization, explicitly required by MS2-INFER-01.
    return log_probability / (((5 + max(length, 1)) / 6) ** alpha)
