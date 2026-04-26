from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MS1Paths:
    repo_root: Path
    data_external: Path
    data_interim: Path
    data_processed_ms1: Path
    experiments_ms1: Path
    docs_reports: Path

    def as_relative_manifest(self) -> dict[str, str]:
        return {
            "data_external": _to_relative(self.data_external, self.repo_root),
            "data_interim": _to_relative(self.data_interim, self.repo_root),
            "data_processed_ms1": _to_relative(self.data_processed_ms1, self.repo_root),
            "experiments_ms1": _to_relative(self.experiments_ms1, self.repo_root),
            "docs_reports": _to_relative(self.docs_reports, self.repo_root),
        }


@dataclass(frozen=True)
class MS2Paths:
    repo_root: Path
    data_processed_ms2: Path
    experiments_ms2: Path
    tokenizer_path: Path
    char_vocab_path: Path
    tfrecord_shard_dir: Path
    run_output_dir: Path
    report_dir: Path

    def as_relative_manifest(self) -> dict[str, str]:
        return {
            "data_processed_ms2": _to_relative(
                self.data_processed_ms2, self.repo_root
            ),
            "experiments_ms2": _to_relative(self.experiments_ms2, self.repo_root),
            "tokenizer_path": _to_relative(self.tokenizer_path, self.repo_root),
            "char_vocab_path": _to_relative(self.char_vocab_path, self.repo_root),
            "tfrecord_shard_dir": _to_relative(
                self.tfrecord_shard_dir, self.repo_root
            ),
            "run_output_dir": _to_relative(self.run_output_dir, self.repo_root),
            "report_dir": _to_relative(self.report_dir, self.repo_root),
        }


def resolve_ms1_paths(
    repo_root: Path | None = None, create_dirs: bool = True
) -> MS1Paths:
    root = (repo_root or Path.cwd()).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid repository root: {root}")

    paths = MS1Paths(
        repo_root=root,
        data_external=root / "data" / "external",
        data_interim=root / "data" / "interim",
        data_processed_ms1=root / "data" / "processed" / "ms1",
        experiments_ms1=root / "experiments" / "ms1",
        docs_reports=root / "docs" / "reports",
    )

    if create_dirs:
        _ensure_output_dirs(paths)

    return paths


def resolve_ms2_paths(
    repo_root: Path | None = None,
    create_dirs: bool = True,
    model: str = "model_a",
    seed: int = 13,
    split: str = "train",
) -> MS2Paths:
    root = (repo_root or Path.cwd()).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid repository root: {root}")
    _validate_token(model, "model")
    _validate_token(split, "split")
    if seed < 0:
        raise ValueError("seed must be >= 0")

    data_processed_ms2 = root / "data" / "processed" / "ms2"
    experiments_ms2 = root / "experiments" / "ms2"
    paths = MS2Paths(
        repo_root=root,
        data_processed_ms2=data_processed_ms2,
        experiments_ms2=experiments_ms2,
        tokenizer_path=data_processed_ms2
        / make_ms2_output_filename("tokenizer", "bpe_4k", 1, "model"),
        char_vocab_path=data_processed_ms2
        / make_ms2_output_filename("char", "vocab", 1, "json"),
        tfrecord_shard_dir=data_processed_ms2
        / make_ms2_output_filename("tfrecords", split, 1, "dir"),
        run_output_dir=experiments_ms2 / model / str(seed),
        report_dir=root / "docs" / "fs" / "ms2",
    )

    if create_dirs:
        _ensure_ms2_output_dirs(paths)

    return paths


def make_ms1_output_filename(stage: str, name: str, version: int, ext: str) -> str:
    _validate_token(stage, "stage")
    _validate_token(name, "name")
    if version < 1:
        raise ValueError("version must be >= 1")

    suffix = ext.lstrip(".")
    if not suffix:
        raise ValueError("ext must not be empty")

    return f"ms1_{stage}_{name}_v{version:03d}.{suffix}"


def make_ms2_output_filename(stage: str, name: str, version: int, ext: str) -> str:
    _validate_token(stage, "stage")
    _validate_token(name, "name")
    if version < 1:
        raise ValueError("version must be >= 1")

    suffix = ext.lstrip(".")
    if not suffix:
        raise ValueError("ext must not be empty")

    return f"ms2_{stage}_{name}_v{version:03d}.{suffix}"


def _ensure_output_dirs(paths: MS1Paths) -> None:
    writable_dirs = (
        paths.data_interim,
        paths.data_processed_ms1,
        paths.experiments_ms1,
        paths.docs_reports,
    )
    for directory in writable_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def _ensure_ms2_output_dirs(paths: MS2Paths) -> None:
    writable_dirs = (
        paths.data_processed_ms2,
        paths.experiments_ms2,
        paths.tfrecord_shard_dir,
        paths.run_output_dir,
        paths.report_dir,
    )
    for directory in writable_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def _to_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _validate_token(value: str, token_name: str) -> None:
    if not value:
        raise ValueError(f"{token_name} must not be empty")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_")
    if any(ch not in allowed for ch in value):
        raise ValueError(
            f"{token_name} must be snake_case alphanumeric with underscores"
        )
