from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelAAblationFlags:
    no_film: bool = False
    mean_merge: bool = False
    plain_branch3: bool = False


def flags_for_variant(variant: str) -> ModelAAblationFlags:
    if variant == "no_film":
        return ModelAAblationFlags(no_film=True)
    if variant == "mean_merge":
        return ModelAAblationFlags(mean_merge=True)
    if variant == "plain_branch3":
        return ModelAAblationFlags(plain_branch3=True)
    raise ValueError(f"not a Model A ablation: {variant}")


def build_model_a_ablation(variant: str):
    from src.ms2.models.rnn.model_a import ModelA

    flags = flags_for_variant(variant)
    return ModelA(
        no_film=flags.no_film,
        mean_merge=flags.mean_merge,
        plain_branch3=flags.plain_branch3,
    )
