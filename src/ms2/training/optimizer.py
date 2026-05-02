from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable
from typing import Any


def _default_decay_var_filter(variable_name: str) -> bool:
    lowered = variable_name.lower()
    excluded_tokens = (
        "embedding",
        "layernorm",
        "layer_norm",
        "bias",
        "beta",
        "gamma",
        "norm",
    )
    return not any(token in lowered for token in excluded_tokens)


@dataclass(frozen=True)
class OptimizerBundle:
    optimizer: Any
    decay_var_filter: Callable[[str], bool]
    decay_variables: tuple[str, ...]
    excluded_decay_variables: tuple[str, ...]
    teacher_forcing_ratio: float
    gradient_clip_norm: float
    applies_selective_weight_decay: bool


def build_adamw(
    *,
    learning_rate: float | Any,
    beta_1: float = 0.9,
    beta_2: float = 0.98,
    epsilon: float = 1e-9,
    weight_decay: float = 0.01,
    gradient_clip_norm: float = 1.0,
    use_loss_scale_optimizer: bool = True,
    decay_var_filter: Callable[[str], bool] | None = None,
    tracked_variables: Iterable[Any] | None = None,
) -> OptimizerBundle:
    """Build an AdamW optimizer bundle following the MS2 ADR defaults.

    The return value is a small integration bundle that keeps the constructed
    optimizer object plus a variable-name filter callable for weight-decay
    exclusions (embeddings, layernorm params, and biases).
    Note: The exclusion info is returned for caller-side enforcement (or loop-side)
    until enforced internally by the optimizer itself.
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
        global_clipnorm=gradient_clip_norm,
    )
    if use_loss_scale_optimizer:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    decay_variables: tuple[str, ...] = ()
    excluded_decay_variables: tuple[str, ...] = ()
    applies_selective_weight_decay = False
    if tracked_variables is not None:
        tracked = tuple(tracked_variables)
        decay_variables = tuple(
            variable.name for variable in tracked if var_filter(variable.name)
        )
        excluded_decay_variables = tuple(
            variable.name for variable in tracked if not var_filter(variable.name)
        )
        inner_optimizer = getattr(
            optimizer,
            "inner_optimizer",
            getattr(optimizer, "inner", optimizer),
        )
        exclude_callable = getattr(inner_optimizer, "exclude_from_weight_decay", None)
        if callable(exclude_callable):
            excluded_names = [
                name.split(":", 1)[0] for name in excluded_decay_variables
            ]
            try:
                exclude_callable(var_names=excluded_names)
            except TypeError:
                excluded_vars = [
                    variable for variable in tracked if not var_filter(variable.name)
                ]
                exclude_callable(var_list=excluded_vars)
            applies_selective_weight_decay = True

    return OptimizerBundle(
        optimizer=optimizer,
        decay_var_filter=var_filter,
        decay_variables=decay_variables,
        excluded_decay_variables=excluded_decay_variables,
        teacher_forcing_ratio=1.0,
        gradient_clip_norm=gradient_clip_norm,
        applies_selective_weight_decay=applies_selective_weight_decay,
    )
