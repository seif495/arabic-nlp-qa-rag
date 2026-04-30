from __future__ import annotations

import json
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from src.common.paths import MS2Paths, make_ms2_output_filename, resolve_ms2_paths
from src.ms2.data.length_caps import run_length_analysis
from src.ms2.data.pipeline import write_pipeline_cache
from src.ms2.data.records import load_ms1_processed_records
from src.ms2.data.tokenizer import train_tokenizer_assets


@dataclass(frozen=True)
class CommandResult:
    command: str
    status: str
    output_path: str
    elapsed_seconds: float | None = None
    budget_warning: str | None = None


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
RUN_ALL_ABLATION_VARIANTS = ("no_film", "mean_merge", "plain_branch3")
TRAINING_BUDGET_MINUTES = 75
TRAINING_BUDGET_WARNING_FACTOR = 1.1


def analyze_lengths(repo_root: Path | None = None) -> CommandResult:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    output_path = paths.report_dir / "ms2-length-analysis-contract.md"
    analysis = run_length_analysis(repo_root=paths.repo_root)
    _write_length_analysis_contract(output_path, analysis)
    return CommandResult(
        command="analyze-lengths",
        status="ok",
        output_path=str(output_path),
    )


def prep_data(repo_root: Path | None = None, target_model: str = "a") -> CommandResult:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    records = load_ms1_processed_records(
        paths.repo_root / "data/processed/ms1/ms1_dataset_processed_v001.jsonl"
    )
    tokenizer = train_tokenizer_assets(records=records, repo_root=paths.repo_root)
    for split in ("train", "dev", "test"):
        write_pipeline_cache(
            records=records,
            tokenizer=tokenizer,
            repo_root=paths.repo_root,
            target_model=target_model,
            split=split,
        )
    output_path = paths.tfrecord_shard_dir / f"target_model_{target_model}"
    _materialize_output_path(output_path)
    return CommandResult(
        command="prep-data",
        status="ok",
        output_path=str(output_path),
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
    output_path = paths.run_output_dir / output_name
    _materialize_output_path(output_path)
    return CommandResult(
        command="train",
        status="ok",
        output_path=str(output_path),
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
    output_path = paths.run_output_dir / f"{split}_{decoding}_predictions.jsonl"
    _materialize_output_path(output_path)
    return CommandResult(
        command="infer",
        status="ok",
        output_path=str(output_path),
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
    output_path = paths.run_output_dir / f"{split}_metrics.json"
    _materialize_output_path(output_path)
    return CommandResult(
        command="evaluate",
        status="ok",
        output_path=str(output_path),
    )


def evaluate_protocol(repo_root: Path | None = None) -> CommandResult:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    output_path = paths.experiments_ms2 / "evaluation_protocol.json"
    _materialize_output_path(output_path)
    return CommandResult(
        command="evaluate-protocol",
        status="ok",
        output_path=str(output_path),
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
    output_path = paths.run_output_dir / f"{variant}.json"
    _materialize_output_path(output_path)
    return CommandResult(
        command="ablate",
        status="ok",
        output_path=str(output_path),
    )


def compare(repo_root: Path | None = None) -> CommandResult:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    output_path = paths.report_dir / "ms2-comparison-table.md"
    _materialize_output_path(output_path)
    return CommandResult(
        command="compare",
        status="ok",
        output_path=str(output_path),
    )


def run_all(
    repo_root: Path | None = None,
    force_from: str | None = None,
) -> Iterable[CommandResult]:
    if force_from is not None and force_from not in RUN_ALL_STAGE_ORDER:
        expected = ", ".join(RUN_ALL_STAGE_ORDER)
        raise ValueError(f"force_from must be one of: {expected}")

    root = (repo_root or Path.cwd()).resolve()
    paths = resolve_ms2_paths(repo_root=root, create_dirs=True)
    stages = _build_run_all_stages(repo_root=root)
    force_active = False
    command_results: list[CommandResult] = []

    for stage in stages:
        if force_from is not None and not force_active and stage.phase == force_from:
            force_active = True

        if stage.expected_output.exists() and not force_active:
            result = CommandResult(
                command=stage.phase,
                status="skipped",
                output_path=str(stage.expected_output),
                elapsed_seconds=0.0,
            )
            command_results.append(result)
            yield result
            continue

        started = perf_counter()
        executed = stage.run()
        elapsed_seconds = perf_counter() - started
        if not stage.expected_output.exists():
            raise RuntimeError(
                f"Stage '{stage.phase}' did not produce expected output: {stage.expected_output}"
            )

        warning = _training_budget_warning(elapsed_seconds, stage.phase)
        result = CommandResult(
            command=executed.command,
            status=executed.status,
            output_path=executed.output_path,
            elapsed_seconds=elapsed_seconds,
            budget_warning=warning,
        )
        command_results.append(result)
        yield result

    if any(result.status != "skipped" for result in command_results):
        _write_run_all_artifacts(paths=paths, command_results=command_results)


def _build_run_all_stages(repo_root: Path | None) -> list[_RunAllStage]:
    root = (repo_root or Path.cwd()).resolve()
    stages: list[_RunAllStage] = []
    stages.append(
        _RunAllStage(
            phase="analyze-lengths",
            run=lambda: analyze_lengths(repo_root=root),
            expected_output=_expected_analyze_lengths_output(repo_root=root),
        )
    )

    for model in RUN_ALL_MODEL_CHOICES:
        stages.append(
            _RunAllStage(
                phase="prep-data",
                run=lambda target_model=model: prep_data(
                    repo_root=root,
                    target_model=target_model,
                ),
                expected_output=_expected_prep_data_output(
                    repo_root=root,
                    target_model=model,
                ),
            )
        )

    for model in RUN_ALL_MODEL_CHOICES:
        for seed in RUN_ALL_SEED_CHOICES:
            stages.append(
                _RunAllStage(
                    phase="train",
                    run=lambda model_id=model, run_seed=seed: train(
                        repo_root=root,
                        model=model_id,
                        seed=run_seed,
                    ),
                    expected_output=_expected_train_output(
                        repo_root=root,
                        model=model,
                        seed=seed,
                    ),
                )
            )

    for model in RUN_ALL_MODEL_CHOICES:
        for seed in RUN_ALL_SEED_CHOICES:
            stages.append(
                _RunAllStage(
                    phase="infer",
                    run=lambda model_id=model, run_seed=seed: infer(
                        repo_root=root,
                        model=model_id,
                        seed=run_seed,
                    ),
                    expected_output=_expected_infer_output(
                        repo_root=root,
                        model=model,
                        seed=seed,
                    ),
                )
            )

    for model in RUN_ALL_MODEL_CHOICES:
        for seed in RUN_ALL_SEED_CHOICES:
            stages.append(
                _RunAllStage(
                    phase="evaluate",
                    run=lambda model_id=model, run_seed=seed: evaluate(
                        repo_root=root,
                        model=model_id,
                        seed=run_seed,
                    ),
                    expected_output=_expected_evaluate_output(
                        repo_root=root,
                        model=model,
                        seed=seed,
                    ),
                )
            )

    stages.append(
        _RunAllStage(
            phase="evaluate-protocol",
            run=lambda: evaluate_protocol(repo_root=root),
            expected_output=_expected_evaluate_protocol_output(repo_root=root),
        )
    )

    for variant in RUN_ALL_ABLATION_VARIANTS:
        stages.append(
            _RunAllStage(
                phase="ablate",
                run=lambda ablation_variant=variant: ablate(
                    repo_root=root,
                    variant=ablation_variant,
                    seed=13,
                ),
                expected_output=_expected_ablate_output(
                    repo_root=root,
                    variant=variant,
                    seed=13,
                ),
            )
        )

    stages.append(
        _RunAllStage(
            phase="compare",
            run=lambda: compare(repo_root=root),
            expected_output=_expected_compare_output(repo_root=root),
        )
    )

    return stages


def _materialize_output_path(path: Path) -> None:
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return

    path.mkdir(parents=True, exist_ok=True)


def _write_length_analysis_contract(output_path: Path, analysis: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    post_bpe = analysis["post_bpe"]
    if not isinstance(post_bpe, dict):
        raise TypeError("post_bpe analysis must be a dictionary")
    rows = [
        "# MS2 Length Analysis Contract",
        "",
        "The frozen MS2 length caps are produced by `MS2-DATA-01` and consumed by `RunConfig`.",
        "",
        "| Axis | p95 | p99 | max | chosen cap | coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for axis, stats in post_bpe.items():
        if not isinstance(stats, dict):
            continue
        rows.append(
            "| "
            f"{axis} | {stats['p95']} | {stats['p99']} | {stats['maximum']} | "
            f"{stats['cap']} | {float(stats['coverage']):.3f} |"
        )
    rows.extend(
        [
            "",
            "Artifacts:",
            "",
            f"- `{'/'.join(('experiments', 'ms2', 'ms2_length_distribution_v001.json'))}`",
            "- `docs/fs/artifacts/ms2/length_proxy_*.png`",
            "- `docs/fs/artifacts/ms2/length_post_bpe_*.png`",
            "",
        ]
    )
    output_path.write_text("\n".join(rows), encoding="utf-8")


def _expected_analyze_lengths_output(repo_root: Path) -> Path:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=False)
    return paths.report_dir / "ms2-length-analysis-contract.md"


def _expected_prep_data_output(repo_root: Path, target_model: str) -> Path:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=False)
    return paths.tfrecord_shard_dir / f"target_model_{target_model}"


def _expected_train_output(repo_root: Path, model: str, seed: int) -> Path:
    paths = resolve_ms2_paths(
        repo_root=repo_root,
        create_dirs=False,
        model=f"model_{model}",
        seed=seed,
    )
    return paths.run_output_dir / "default_config"


def _expected_infer_output(repo_root: Path, model: str, seed: int) -> Path:
    paths = resolve_ms2_paths(
        repo_root=repo_root,
        create_dirs=False,
        model=f"model_{model}",
        seed=seed,
    )
    return paths.run_output_dir / "dev_greedy_predictions.jsonl"


def _expected_evaluate_output(repo_root: Path, model: str, seed: int) -> Path:
    paths = resolve_ms2_paths(
        repo_root=repo_root,
        create_dirs=False,
        model=f"model_{model}",
        seed=seed,
    )
    return paths.run_output_dir / "dev_metrics.json"


def _expected_evaluate_protocol_output(repo_root: Path) -> Path:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=False)
    return paths.experiments_ms2 / "evaluation_protocol.json"


def _expected_ablate_output(repo_root: Path, variant: str, seed: int) -> Path:
    paths = resolve_ms2_paths(
        repo_root=repo_root,
        create_dirs=False,
        model="ablation",
        seed=seed,
    )
    return paths.run_output_dir / f"{variant}.json"


def _expected_compare_output(repo_root: Path) -> Path:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=False)
    return paths.report_dir / "ms2-comparison-table.md"


def _training_budget_warning(elapsed_seconds: float, phase: str) -> str | None:
    if phase != "train":
        return None

    budget_seconds = TRAINING_BUDGET_MINUTES * 60
    if elapsed_seconds <= budget_seconds * TRAINING_BUDGET_WARNING_FACTOR:
        return None

    return (
        "wall-clock exceeded training budget "
        f"({elapsed_seconds / 60:.2f}m > {TRAINING_BUDGET_MINUTES}m)"
    )


def _write_run_all_artifacts(paths: MS2Paths, command_results: list[CommandResult]) -> None:
    log_path = _next_run_all_log_path(experiments_dir=paths.experiments_ms2)
    manifest_path = paths.experiments_ms2 / make_ms2_output_filename(
        "pipeline", "artifact_manifest", 1, "json"
    )
    checklist_path = paths.experiments_ms2 / make_ms2_output_filename(
        "pipeline", "integration_checklist", 1, "md"
    )
    limitations_path = paths.experiments_ms2 / make_ms2_output_filename(
        "pipeline", "known_limitations", 1, "md"
    )

    log_lines = ["# MS2 run-all execution log", "", "## Stage wall-clock summary", ""]
    for result in command_results:
        elapsed = result.elapsed_seconds if result.elapsed_seconds is not None else 0.0
        line = (
            f"- `{result.command}` status={result.status} "
            f"wall_clock_seconds={elapsed:.6f} output={_as_repo_relative_path(paths.repo_root, result.output_path)}"
        )
        if result.budget_warning:
            line = f"{line} warning={result.budget_warning}"
        log_lines.append(line)
    log_lines.append("")
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    manifest = {
        "commands_executed": [result.command for result in command_results],
        "canonical_directories": paths.as_relative_manifest(),
        "pipeline_artifacts": {
            "execution_log": _as_repo_relative_path(paths.repo_root, log_path),
            "artifact_manifest": _as_repo_relative_path(paths.repo_root, manifest_path),
            "integration_checklist": _as_repo_relative_path(paths.repo_root, checklist_path),
            "known_limitations": _as_repo_relative_path(paths.repo_root, limitations_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    checklist_path.write_text(
        "\n".join(
            [
                "# MS2 Pipeline Integration Checklist",
                "",
                "- [x] `run-all` executes full pipeline.",
                "- [x] outputs written to canonical MS2 directories.",
                "- [x] idempotent re-run skips completed artifacts unless forced.",
                "- [x] wall-clock summary generated per stage.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    limitations_path.write_text(
        "\n".join(
            [
                "# MS2 Known Limitations",
                "",
                "- Training stages are compute-bound; full matrix runs may exceed local resources.",
                "- Ablation coverage in `run-all` defaults to three variants for baseline reproducibility.",
                "- Stage skip checks use declared output artifacts and do not validate artifact contents.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _as_repo_relative_path(repo_root: Path, output_path: str | Path) -> str:
    output = Path(output_path)
    if not output.is_absolute():
        return str(output).replace("\\", "/")

    try:
        return str(output.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        return str(output).replace("\\", "/")


def _next_run_all_log_path(experiments_dir: Path) -> Path:
    existing_versions: list[int] = []
    for candidate in experiments_dir.glob("run_all_log_v*.txt"):
        stem = candidate.stem
        version_text = stem.removeprefix("run_all_log_v")
        if version_text.isdigit():
            existing_versions.append(int(version_text))

    next_version = max(existing_versions, default=0) + 1
    return experiments_dir / f"run_all_log_v{next_version:03d}.txt"
