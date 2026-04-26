from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


def configure_runtime(
    seed: int, mixed_precision: bool = True, deterministic: bool = True
) -> dict[str, Any]:
    """Configure MS2 TensorFlow/Keras reproducibility and numeric policy.

    This centralizes ADR §1.5 mixed precision and ADR §1.9 seeded deterministic
    execution so downstream MS2 entrypoints call it exactly once at process start.
    """
    if seed < 0:
        raise ValueError("seed must be >= 0")

    try:
        import tensorflow as tf  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without TF installed
        raise RuntimeError(
            "TensorFlow is required to configure the MS2 runtime."
        ) from exc

    policy_name = "mixed_float16" if mixed_precision else "float32"
    tf.keras.mixed_precision.set_global_policy(policy_name)
    tf.keras.utils.set_random_seed(seed)

    determinism_enabled = False
    if deterministic:
        tf.config.experimental.enable_op_determinism()
        determinism_enabled = True

    summary = {
        "seed": seed,
        "mixed_precision_policy": policy_name,
        "deterministic_ops": determinism_enabled,
    }
    LOGGER.info(
        "Configured MS2 runtime: seed=%s mixed_precision_policy=%s deterministic_ops=%s",
        summary["seed"],
        summary["mixed_precision_policy"],
        summary["deterministic_ops"],
    )
    return summary
