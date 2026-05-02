from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.ms2.inference.greedy import DecodableModel, DecodeResult, _length_normalized


@dataclass(frozen=True)
class _Beam:
    tokens: tuple[int, ...]
    log_probability: float
    state: Any
    ended: bool


def beam_decode(
    model: DecodableModel,
    encoder_inputs: Any,
    bos_id: int = 2,
    eos_id: int = 3,
    max_length: int = 64,
    beam_width: int = 4,
) -> DecodeResult:
    """Beam-4 decode with `((5+L)/6)^0.6` length normalization."""
    beams = [
        _Beam(
            tokens=(),
            log_probability=0.0,
            state=model.init_state(encoder_inputs),
            ended=False,
        )
    ]
    for _ in range(max_length):
        candidates: list[_Beam] = []
        for beam in beams:
            if beam.ended:
                candidates.append(beam)
                continue
            last = beam.tokens[-1] if beam.tokens else bos_id
            logits, state = model.step(beam.state, last)
            top_ids = sorted(
                range(len(logits)), key=lambda idx: logits[idx], reverse=True
            )[:beam_width]
            for token_id in top_ids:
                tokens = (*beam.tokens, token_id)
                candidates.append(
                    _Beam(
                        tokens=tokens,
                        log_probability=beam.log_probability + float(logits[token_id]),
                        state=state,
                        ended=token_id == eos_id,
                    )
                )
        beams = sorted(
            candidates,
            key=lambda b: _length_normalized(b.log_probability, len(b.tokens)),
            reverse=True,
        )[:beam_width]
        if all(beam.ended for beam in beams):
            break
    best = max(
        beams, key=lambda b: _length_normalized(b.log_probability, len(b.tokens))
    )
    return DecodeResult(
        list(best.tokens),
        best.log_probability,
        _length_normalized(best.log_probability, len(best.tokens)),
    )
