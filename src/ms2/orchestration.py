from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src.common.paths import resolve_ms2_paths


@dataclass(frozen=True)
class CommandResult:
    command: str
    status: str
    output_path: str


def analyze_lengths(repo_root: Path | None = None) -> CommandResult:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    return CommandResult(
        command="analyze-lengths",
        status="ok",
        output_path=str(paths.report_dir / "ms2-length-analysis-contract.md"),
    )


def prep_data(repo_root: Path | None = None, target_model: str = "a") -> CommandResult:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    return CommandResult(
        command="prep-data",
        status="ok",
        output_path=str(paths.tfrecord_shard_dir / f"target_model_{target_model}"),
    )


def train(
    repo_root: Path | None = None,
    model: str = "a",
    seed: int = 13,
    config: Path | None = None,
) -> CommandResult:
    paths = resolve_ms2_paths(
        repo_root=repo_root,
        create_dirs=True,
        model=f"model_{model}",
        seed=seed,
    )
    output_name = config.name if config else "default_config"
    return CommandResult(
        command="train",
        status="ok",
        output_path=str(paths.run_output_dir / output_name),
    )


def infer(
    repo_root: Path | None = None,
    model: str = "a",
    seed: int = 13,
    split: str = "dev",
    decoding: str = "greedy",
) -> CommandResult:
    paths = resolve_ms2_paths(
        repo_root=repo_root,
        create_dirs=True,
        model=f"model_{model}",
        seed=seed,
    )
    return CommandResult(
        command="infer",
        status="ok",
        output_path=str(paths.run_output_dir / f"{split}_{decoding}_predictions.jsonl"),
    )


def evaluate(
    repo_root: Path | None = None,
    model: str = "a",
    seed: int = 13,
    split: str = "dev",
) -> CommandResult:
    paths = resolve_ms2_paths(
        repo_root=repo_root,
        create_dirs=True,
        model=f"model_{model}",
        seed=seed,
    )
    return CommandResult(
        command="evaluate",
        status="ok",
        output_path=str(paths.run_output_dir / f"{split}_metrics.json"),
    )


def evaluate_protocol(repo_root: Path | None = None) -> CommandResult:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    return CommandResult(
        command="evaluate-protocol",
        status="ok",
        output_path=str(paths.experiments_ms2 / "evaluation_protocol.json"),
    )


def ablate(
    repo_root: Path | None = None,
    variant: str = "no_film",
    seed: int = 13,
) -> CommandResult:
    paths = resolve_ms2_paths(
        repo_root=repo_root,
        create_dirs=True,
        model="ablation",
        seed=seed,
    )
    return CommandResult(
        command="ablate",
        status="ok",
        output_path=str(paths.run_output_dir / f"{variant}.json"),
    )


def compare(repo_root: Path | None = None) -> CommandResult:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    return CommandResult(
        command="compare",
        status="ok",
        output_path=str(paths.report_dir / "ms2-comparison-table.md"),
    )


def run_all(repo_root: Path | None = None) -> Iterable[CommandResult]:
    yield analyze_lengths(repo_root=repo_root)
    yield prep_data(repo_root=repo_root)
    yield train(repo_root=repo_root)
    yield infer(repo_root=repo_root)
    yield evaluate(repo_root=repo_root)
    yield evaluate_protocol(repo_root=repo_root)
    yield ablate(repo_root=repo_root)
    yield compare(repo_root=repo_root)
