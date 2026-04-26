from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src.common.paths import resolve_ms2_paths


@dataclass(frozen=True)
class CommandResult:
    command: str
    status: str
    output_path: str


@dataclass(frozen=True)
class _RunAllStage:
    phase: str
    run: Callable[[], CommandResult]
    expected_output: Path


RUN_ALL_STAGE_ORDER = (
    "analyze-lengths",
    "prep-data",
    "train",
    "infer",
    "evaluate",
    "evaluate-protocol",
    "ablate",
    "compare",
)
RUN_ALL_MODEL_CHOICES = ("a", "b")
RUN_ALL_SEED_CHOICES = (13, 42, 91)


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


def run_all(
    repo_root: Path | None = None,
    force_from: str | None = None,
) -> Iterable[CommandResult]:
    if force_from is not None and force_from not in RUN_ALL_STAGE_ORDER:
        expected = ", ".join(RUN_ALL_STAGE_ORDER)
        raise ValueError(f"force_from must be one of: {expected}")

    stages = _build_run_all_stages(repo_root=repo_root)
    force_active = force_from is None

    for stage in stages:
        if not force_active and stage.phase == force_from:
            force_active = True

        if stage.expected_output.exists() and not force_active:
            yield CommandResult(
                command=stage.phase,
                status="skipped",
                output_path=str(stage.expected_output),
            )
            continue

        yield stage.run()


def _build_run_all_stages(repo_root: Path | None) -> list[_RunAllStage]:
    stages: list[_RunAllStage] = []
    stages.append(
        _RunAllStage(
            phase="analyze-lengths",
            run=lambda: analyze_lengths(repo_root=repo_root),
            expected_output=Path(analyze_lengths(repo_root=repo_root).output_path),
        )
    )

    for model in RUN_ALL_MODEL_CHOICES:
        stages.append(
            _RunAllStage(
                phase="prep-data",
                run=lambda target_model=model: prep_data(
                    repo_root=repo_root,
                    target_model=target_model,
                ),
                expected_output=Path(
                    prep_data(
                        repo_root=repo_root,
                        target_model=model,
                    ).output_path
                ),
            )
        )

    for model in RUN_ALL_MODEL_CHOICES:
        for seed in RUN_ALL_SEED_CHOICES:
            stages.append(
                _RunAllStage(
                    phase="train",
                    run=lambda model_id=model, run_seed=seed: train(
                        repo_root=repo_root,
                        model=model_id,
                        seed=run_seed,
                    ),
                    expected_output=Path(
                        train(
                            repo_root=repo_root,
                            model=model,
                            seed=seed,
                        ).output_path
                    ),
                )
            )

    for model in RUN_ALL_MODEL_CHOICES:
        for seed in RUN_ALL_SEED_CHOICES:
            stages.append(
                _RunAllStage(
                    phase="infer",
                    run=lambda model_id=model, run_seed=seed: infer(
                        repo_root=repo_root,
                        model=model_id,
                        seed=run_seed,
                    ),
                    expected_output=Path(
                        infer(
                            repo_root=repo_root,
                            model=model,
                            seed=seed,
                        ).output_path
                    ),
                )
            )

    for model in RUN_ALL_MODEL_CHOICES:
        for seed in RUN_ALL_SEED_CHOICES:
            stages.append(
                _RunAllStage(
                    phase="evaluate",
                    run=lambda model_id=model, run_seed=seed: evaluate(
                        repo_root=repo_root,
                        model=model_id,
                        seed=run_seed,
                    ),
                    expected_output=Path(
                        evaluate(
                            repo_root=repo_root,
                            model=model,
                            seed=seed,
                        ).output_path
                    ),
                )
            )

    stages.append(
        _RunAllStage(
            phase="evaluate-protocol",
            run=lambda: evaluate_protocol(repo_root=repo_root),
            expected_output=Path(evaluate_protocol(repo_root=repo_root).output_path),
        )
    )

    for variant in ("no_film", "mean_merge", "plain_branch3"):
        stages.append(
            _RunAllStage(
                phase="ablate",
                run=lambda ablation_variant=variant: ablate(
                    repo_root=repo_root,
                    variant=ablation_variant,
                    seed=13,
                ),
                expected_output=Path(
                    ablate(
                        repo_root=repo_root,
                        variant=variant,
                        seed=13,
                    ).output_path
                ),
            )
        )

    stages.append(
        _RunAllStage(
            phase="compare",
            run=lambda: compare(repo_root=repo_root),
            expected_output=Path(compare(repo_root=repo_root).output_path),
        )
    )

    return stages
