from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelBAblationFlags:
    positional_mode: str = "rope"
    shared_layers: bool = False


def flags_for_variant(variant: str) -> ModelBAblationFlags:
    if variant == "sinusoidal_pe":
        return ModelBAblationFlags(positional_mode="sinusoidal_pe")
    if variant == "no_pe":
        return ModelBAblationFlags(positional_mode="no_pe")
    if variant == "shared_layers":
        return ModelBAblationFlags(shared_layers=True)
    raise ValueError(f"not a Model B ablation: {variant}")


def build_model_b_ablation(variant: str):
    from src.ms2.models.transformer.model_b import ModelB

    flags = flags_for_variant(variant)
    return ModelB(
        positional_mode=flags.positional_mode, shared_layers=flags.shared_layers
    )
