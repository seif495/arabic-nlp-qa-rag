from __future__ import annotations

import math
from typing import Any

try:
    import tensorflow as tf  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    tf = None  # type: ignore[assignment]


if tf is not None:
    _ScheduleBase = tf.keras.optimizers.schedules.LearningRateSchedule
else:  # pragma: no cover
    class _ScheduleBase:
        pass


class CosineWithWarmup(_ScheduleBase):
    def __init__(
        self,
        total_steps: int,
        peak_lr: float = 3e-4,
        min_lr: float = 1e-5,
        warmup_ratio: float = 0.05,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if peak_lr <= 0 or min_lr <= 0:
            raise ValueError("peak_lr and min_lr must be positive")
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = max(1, int(total_steps * warmup_ratio))

    def __call__(self, step: int | float) -> float:
        step_value = float(step)
        if step_value <= self.warmup_steps:
            return self.peak_lr * (step_value / float(self.warmup_steps))
        if step_value >= self.total_steps:
            return self.min_lr
        decay_steps = self.total_steps - self.warmup_steps
        progress = (step_value - self.warmup_steps) / float(decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def get_config(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "warmup_ratio": self.warmup_ratio,
        }


class Noam(_ScheduleBase):
    def __init__(self, d_model: int, warmup_steps: int = 1000) -> None:
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step: int | float) -> float:
        step_value = max(float(step), 1.0)
        scale = self.d_model**-0.5
        return scale * min(step_value**-0.5, step_value * self.warmup_steps**-1.5)

    def get_config(self) -> dict[str, Any]:
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}
