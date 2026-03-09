from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.paths import resolve_ms1_paths
from src.ms1.dataset_export import export_processed_dataset


@dataclass(frozen=True)
class CommandResult:
    command: str
    status: str
    output_path: str


def profile(repo_root: Path | None = None) -> CommandResult:
    paths = resolve_ms1_paths(repo_root=repo_root)
    return CommandResult(
        command="profile",
        status="ok",
        output_path=str(paths.experiments_ms1 / "profile"),
    )


def detect_irregularities(repo_root: Path | None = None) -> CommandResult:
    paths = resolve_ms1_paths(repo_root=repo_root)
    return CommandResult(
        command="detect-irregularities",
        status="ok",
        output_path=str(paths.experiments_ms1 / "irregularities"),
    )


def normalize(repo_root: Path | None = None) -> CommandResult:
    paths = resolve_ms1_paths(repo_root=repo_root)
    return CommandResult(
        command="normalize",
        status="ok",
        output_path=str(paths.data_interim / "normalized"),
    )


def build_dataset(repo_root: Path | None = None) -> CommandResult:
    paths = resolve_ms1_paths(repo_root=repo_root)
    export_processed_dataset(paths=paths)
    return CommandResult(
        command="build-dataset",
        status="ok",
        output_path=str(paths.data_processed_ms1),
    )


def run_all(repo_root: Path | None = None) -> list[CommandResult]:
    return [
        profile(repo_root=repo_root),
        detect_irregularities(repo_root=repo_root),
        normalize(repo_root=repo_root),
        build_dataset(repo_root=repo_root),
    ]
