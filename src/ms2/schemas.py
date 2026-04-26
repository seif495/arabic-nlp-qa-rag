from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, TypeVar, get_args

ModelId = Literal["A", "B"]
LRSchedule = Literal["cosine_with_warmup", "noam"]
AblationVariant = Literal[
    "no_film",
    "mean_merge",
    "plain_branch3",
    "sinusoidal_pe",
    "no_pe",
    "shared_layers",
]

_T = TypeVar("_T")


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze_json_value(value[key])) for key in sorted(value))
    if isinstance(value, list | tuple):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _coerce_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list | tuple):
        raise TypeError("expected a JSON array for tuple field")
    return tuple(int(item) for item in value)


def _require_literal(value: str, allowed: tuple[str, ...], field_name: str) -> str:
    if value not in allowed:
        expected = ", ".join(allowed)
        raise ValueError(f"{field_name} must be one of: {expected}")
    return value


@dataclass(frozen=True)
class RunConfig:
    """Frozen-at-launch MS2 training run configuration.

    ADR sources: `model_id` and length caps (`l_q`, `l_c`, `l_enc`, `l_dec`)
    come from ADR §1.3; `bucket_boundaries` and `target_tokens_per_batch`
    come from ADR §1.4; `label_smoothing` comes from ADR §1.7; `seed` comes
    from ADR §1.9; `lr_schedule`, `wall_clock_budget_minutes`, and
    `gradient_clip_norm` come from ADR §2.12 / §3.7 and the matched-compute
    budget in ADR §4.1.
    """

    model_id: ModelId
    seed: int
    l_q: int = 32
    l_c: int = 384
    l_enc: int = 420
    l_dec: int = 64
    bucket_boundaries: tuple[int, ...] = (128, 192, 256, 320, 420)
    target_tokens_per_batch: int = 16_384
    lr_schedule: LRSchedule = "cosine_with_warmup"
    wall_clock_budget_minutes: int = 75
    label_smoothing: float = 0.1
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        _require_literal(self.model_id, get_args(ModelId), "model_id")
        _require_literal(self.lr_schedule, get_args(LRSchedule), "lr_schedule")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if any(value <= 0 for value in (self.l_q, self.l_c, self.l_enc, self.l_dec)):
            raise ValueError("length caps must be positive")
        if tuple(sorted(self.bucket_boundaries)) != self.bucket_boundaries:
            raise ValueError("bucket_boundaries must be sorted ascending")
        if self.target_tokens_per_batch <= 0:
            raise ValueError("target_tokens_per_batch must be positive")
        if self.wall_clock_budget_minutes <= 0:
            raise ValueError("wall_clock_budget_minutes must be positive")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        values = dict(data)
        if "bucket_boundaries" in values:
            values["bucket_boundaries"] = _coerce_tuple(values["bucket_boundaries"])
        return cls(**values)

    def frozen_dict(self) -> tuple[tuple[str, Any], ...]:
        """Return a hashable view of every binding field for run identity checks."""
        return _freeze_json_value(self.to_dict())


@dataclass(frozen=True)
class BatchSpec:
    """Declarative MS2 batch tensor contract.

    ADR sources: encoder/decoder lengths come from ADR §1.3; `padding_id=0`
    comes from ADR §1.1; `char_matrix_shape` and `l_char_max=16` come from
    ADR §2.6.
    """

    batch_size: int
    encoder_input_shape: tuple[int, int]
    decoder_input_shape: tuple[int, int]
    decoder_target_shape: tuple[int, int]
    char_matrix_shape: tuple[int, int, int]
    padding_id: int = 0
    l_char_max: int = 16

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for field_name in (
            "encoder_input_shape",
            "decoder_input_shape",
            "decoder_target_shape",
            "char_matrix_shape",
        ):
            shape = getattr(self, field_name)
            if any(dim <= 0 for dim in shape):
                raise ValueError(f"{field_name} dimensions must be positive")
        if self.padding_id != 0:
            raise ValueError("padding_id must be 0")
        if self.l_char_max != 16:
            raise ValueError("l_char_max must be 16")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchSpec:
        values = dict(data)
        for field_name in (
            "encoder_input_shape",
            "decoder_input_shape",
            "decoder_target_shape",
            "char_matrix_shape",
        ):
            if field_name in values:
                values[field_name] = _coerce_tuple(values[field_name])
        return cls(**values)


@dataclass(frozen=True)
class RunSummary:
    """Result-side metrics and resource summary for one completed MS2 run.

    ADR sources: EM, Token-F1, normalized character edit distance, and BLEU-1
    come from ADR §1.8; parameter count, peak GPU memory, train wall-clock,
    and mean inference time per example feed the comparison protocol in ADR
    §4.1 and §4.3.
    """

    run_id: str
    model_id: ModelId
    seed: int
    dev_em: float
    dev_token_f1: float
    dev_char_edit_distance: float
    dev_bleu1: float
    test_em: float
    test_token_f1: float
    test_char_edit_distance: float
    test_bleu1: float
    parameter_count: int
    peak_gpu_memory_mb: float
    train_wall_clock_minutes: float
    mean_inference_time_ms_per_example: float

    def __post_init__(self) -> None:
        _require_literal(self.model_id, get_args(ModelId), "model_id")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.parameter_count <= 0:
            raise ValueError("parameter_count must be positive")
        for field_def in fields(self):
            if field_def.name.endswith(("em", "f1", "distance", "bleu1")):
                value = getattr(self, field_def.name)
                if not 0 <= value <= 1:
                    raise ValueError(f"{field_def.name} must be in [0, 1]")
        if self.peak_gpu_memory_mb < 0:
            raise ValueError("peak_gpu_memory_mb must be >= 0")
        if self.train_wall_clock_minutes < 0:
            raise ValueError("train_wall_clock_minutes must be >= 0")
        if self.mean_inference_time_ms_per_example < 0:
            raise ValueError("mean_inference_time_ms_per_example must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunSummary:
        return cls(**data)


@dataclass(frozen=True)
class AblationConfig:
    """Lightweight MS2 ablation configuration over a base run.

    ADR sources: `base_run_config` is the frozen training contract from ADR
    §1.3, §1.4, §1.7, §1.9, §2.12, §3.7, and §4.1; `variant` enumerates the
    ablation switches named in the MS2 comparison/ablation plan.
    """

    base_run_config: RunConfig
    variant: AblationVariant

    def __post_init__(self) -> None:
        _require_literal(self.variant, get_args(AblationVariant), "variant")

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_run_config": self.base_run_config.to_dict(),
            "variant": self.variant,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AblationConfig:
        values = dict(data)
        values["base_run_config"] = RunConfig.from_dict(values["base_run_config"])
        return cls(**values)


EXAMPLE_RUN_CONFIG = RunConfig(model_id="A", seed=13)

EXAMPLE_BATCH_SPEC = BatchSpec(
    batch_size=32,
    encoder_input_shape=(32, 420),
    decoder_input_shape=(32, 64),
    decoder_target_shape=(32, 64),
    char_matrix_shape=(32, 420, 16),
)

EXAMPLE_RUN_SUMMARY = RunSummary(
    run_id="model_a_seed_13",
    model_id="A",
    seed=13,
    dev_em=0.42,
    dev_token_f1=0.61,
    dev_char_edit_distance=0.18,
    dev_bleu1=0.55,
    test_em=0.4,
    test_token_f1=0.59,
    test_char_edit_distance=0.2,
    test_bleu1=0.53,
    parameter_count=2_700_000,
    peak_gpu_memory_mb=6144.0,
    train_wall_clock_minutes=75.0,
    mean_inference_time_ms_per_example=18.5,
)

EXAMPLE_ABLATION_CONFIG = AblationConfig(
    base_run_config=EXAMPLE_RUN_CONFIG,
    variant="no_film",
)
