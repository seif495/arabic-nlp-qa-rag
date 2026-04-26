from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ms2.schemas import (
    EXAMPLE_ABLATION_CONFIG,
    EXAMPLE_BATCH_SPEC,
    EXAMPLE_RUN_CONFIG,
    EXAMPLE_RUN_SUMMARY,
    AblationConfig,
    BatchSpec,
    RunConfig,
    RunSummary,
)


@pytest.mark.parametrize(
    ("schema_type", "example"),
    [
        (RunConfig, EXAMPLE_RUN_CONFIG),
        (BatchSpec, EXAMPLE_BATCH_SPEC),
        (RunSummary, EXAMPLE_RUN_SUMMARY),
        (AblationConfig, EXAMPLE_ABLATION_CONFIG),
    ],
)
def test_ms2_schemas_round_trip_through_json(
    schema_type: type[RunConfig | BatchSpec | RunSummary | AblationConfig],
    example: RunConfig | BatchSpec | RunSummary | AblationConfig,
) -> None:
    payload = json.loads(json.dumps(example.to_dict()))

    assert schema_type.from_dict(payload) == example


def test_run_config_frozen_dict_is_hashable_and_field_sensitive() -> None:
    base = RunConfig(model_id="A", seed=13)
    same = RunConfig(model_id="A", seed=13)
    changed = RunConfig(model_id="A", seed=42)

    assert hash(base.frozen_dict()) == hash(same.frozen_dict())
    assert base.frozen_dict() != changed.frozen_dict()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"encoder_input_shape": (31, 420)},
        {"decoder_input_shape": (31, 64)},
        {"decoder_target_shape": (31, 64)},
        {"char_matrix_shape": (31, 420, 16)},
        {"char_matrix_shape": (32, 420, 8)},
    ],
)
def test_batch_spec_rejects_inconsistent_shape_contract(
    kwargs: dict[str, tuple[int, ...]],
) -> None:
    values = EXAMPLE_BATCH_SPEC.to_dict()
    values.update(kwargs)

    with pytest.raises(ValueError):
        BatchSpec.from_dict(values)


@pytest.mark.parametrize(
    ("artifact_name", "schema_type", "example"),
    [
        ("ms2-run-config.example.json", RunConfig, EXAMPLE_RUN_CONFIG),
        ("ms2-batch-spec.example.json", BatchSpec, EXAMPLE_BATCH_SPEC),
        ("ms2-run-summary.example.json", RunSummary, EXAMPLE_RUN_SUMMARY),
        ("ms2-ablation-config.example.json", AblationConfig, EXAMPLE_ABLATION_CONFIG),
    ],
)
def test_ms2_schema_json_examples_are_in_sync(
    artifact_name: str,
    schema_type: type[RunConfig | BatchSpec | RunSummary | AblationConfig],
    example: RunConfig | BatchSpec | RunSummary | AblationConfig,
) -> None:
    artifact_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "fs"
        / "artifacts"
        / "ms2"
        / artifact_name
    )

    with artifact_path.open(encoding="utf-8") as file_obj:
        artifact_payload = json.load(file_obj)

    assert schema_type.from_dict(artifact_payload) == example


def test_schema_contract_mentions_each_schema() -> None:
    contract_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "fs"
        / "ms2"
        / "ms2-schema-contract.md"
    )
    contract = contract_path.read_text(encoding="utf-8")

    for schema_name in ("RunConfig", "BatchSpec", "RunSummary", "AblationConfig"):
        assert f"## {schema_name}" in contract
