from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def _default_decay_var_filter(variable_name: str) -> bool:
    lowered = variable_name.lower()
    excluded_tokens = ("embedding", "layernorm", "bias")
    return not any(token in lowered for token in excluded_tokens)


def build_adamw(
    *,
    learning_rate: float | Any,
    beta_1: float = 0.9,
    beta_2: float = 0.98,
    epsilon: float = 1e-9,
    weight_decay: float = 0.01,
    use_loss_scale_optimizer: bool = True,
    decay_var_filter: Callable[[str], bool] | None = None,
    tracked_variables: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Build an AdamW optimizer bundle following the MS2 ADR defaults.

    The return value is a small integration bundle that keeps the constructed
    optimizer object plus a variable-name filter callable for weight-decay
    exclusions (embeddings, layernorm params, and biases).
    """

    try:
        import tensorflow as tf  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required to build MS2 optimizers.") from exc

    var_filter = decay_var_filter or _default_decay_var_filter
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        weight_decay=weight_decay,
    )
    if use_loss_scale_optimizer:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    decay_variables: tuple[str, ...] = ()
    if tracked_variables is not None:
        decay_variables = tuple(
            variable.name for variable in tracked_variables if var_filter(variable.name)
        )

    return {
        "optimizer": optimizer,
        "decay_var_filter": var_filter,
        "decay_variables": decay_variables,
        "teacher_forcing_ratio": 1.0,
    }
