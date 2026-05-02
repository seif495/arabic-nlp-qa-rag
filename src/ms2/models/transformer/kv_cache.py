from __future__ import annotations

from dataclasses import dataclass, field

import tensorflow as tf


@dataclass
class KVCache:
    """Small autoregressive K/V cache for transformer decoding, ADR §3.8."""

    self_attention: list[dict[str, tf.Tensor]] = field(default_factory=list)
    cross_attention: list[dict[str, tf.Tensor]] = field(default_factory=list)

    @classmethod
    def for_layers(cls, n_layers: int) -> KVCache:
        return cls(
            self_attention=[{} for _ in range(n_layers)],
            cross_attention=[{} for _ in range(n_layers)],
        )

    def populated_steps(self) -> int:
        if not self.self_attention or "k" not in self.self_attention[0]:
            return 0
        steps = self.self_attention[0]["k"].shape[2]
        if steps is not None:
            return int(steps)
        return int(tf.shape(self.self_attention[0]["k"])[2].numpy())
