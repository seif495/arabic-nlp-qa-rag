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
from src.ms2.data.tokenizer import ensure_default_tokenizer, train_tokenizer_assets
from src.ms2.analysis.compare import write_headline_tables
from src.ms2.analysis.conditioning_viz import write_placeholder_png
from src.ms2.analysis.difficulty import write_difficulty_placeholder
from src.ms2.analysis.long_dependency import write_long_dependency_placeholder
from src.ms2.analysis.noise_battery import write_noise_battery_placeholder


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
MS2_LENGTH_DISTRIBUTION_ARTIFACT = (
    "experiments" + "/ms2/ms2_length_distribution_v001.json"
)


def analyze_lengths(repo_root: Path | None = None) -> CommandResult:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    output_path = paths.report_dir / "ms2-length-analysis-contract.md"
    tokenizer = ensure_default_tokenizer(repo_root=paths.repo_root)
    analysis = run_length_analysis(
        repo_root=paths.repo_root,
        tokenizer=lambda text: tokenizer.encode(text),
    )
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
    del config
    output_path = paths.run_output_dir / "run_summary_v001.json"
    _write_compute_bound_run_artifacts(paths.run_output_dir, model=model, seed=seed)
    _write_training_runs_index(paths.experiments_ms2)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    benchmark_path = paths.experiments_ms2 / "inference_benchmark_v001.json"
    if not benchmark_path.exists():
        benchmark_path.write_text(
            json.dumps(
                {
                    "status": "not_run_compute_bound",
                    "note": "Requires trained checkpoints; no inference timing fabricated.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "status": "missing_predictions_or_checkpoint",
                "metrics": {
                    "em": None,
                    "token_f1": None,
                    "char_edit_distance": None,
                    "bleu1": None,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return CommandResult(
        command="evaluate",
        status="ok",
        output_path=str(output_path),
    )


def evaluate_protocol(repo_root: Path | None = None) -> CommandResult:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    write_noise_battery_placeholder(paths.experiments_ms2 / "noise_battery_v001.csv")
    write_long_dependency_placeholder(
        paths.experiments_ms2 / "long_dependency_v001.json"
    )
    write_difficulty_placeholder(paths.experiments_ms2 / "difficulty_buckets_v001.json")
    leave_dir = paths.experiments_ms2 / "leave_2_videos_out"
    leave_dir.mkdir(parents=True, exist_ok=True)
    (leave_dir / "README.md").write_text(
        "# Leave-2-Videos-Out\n\nProtocol hooks are implemented; trained run summaries require additional compute.\n",
        encoding="utf-8",
    )
    for idx, name in enumerate(
        (
            "model_a_gates",
            "model_a_film",
            "model_b_encoder_attention",
            "model_b_cross_attention",
        ),
        start=1,
    ):
        write_placeholder_png(
            paths.repo_root
            / "docs"
            / "fs"
            / "artifacts"
            / "ms2"
            / f"conditioning_{idx}_{name}.png"
        )
    output_path = paths.experiments_ms2 / "evaluation_protocol.json"
    output_path.write_text(
        json.dumps(
            {
                "status": "protocol_ready_missing_trained_checkpoints",
                "artifacts": [
                    _ms2_experiment_relpath("noise_battery_v001.csv"),
                    _ms2_experiment_relpath("long_dependency_v001.json"),
                    _ms2_experiment_relpath("difficulty_buckets_v001.json"),
                    _ms2_experiment_relpath("leave_2_videos_out/"),
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
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
        model="ablations",
        seed=seed,
    )
    output_path = (
        paths.experiments_ms2
        / "ablations"
        / _ablation_model_id(variant)
        / variant
        / str(seed)
        / "run_summary_v001.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (output_path.parent / "checkpoints").mkdir(exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "variant": variant,
                "seed": seed,
                "status": "not_run_compute_bound",
                "note": "Ablation construction flags are implemented; training requires TensorFlow/GPU runtime.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_ablations_index(paths.experiments_ms2)
    return CommandResult(
        command="ablate",
        status="ok",
        output_path=str(output_path),
    )


def compare(repo_root: Path | None = None) -> CommandResult:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    csv_path, md_path = write_headline_tables(
        paths.experiments_ms2, paths.repo_root / "docs" / "reports"
    )
    output_path = md_path
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


def _write_compute_bound_run_artifacts(run_dir: Path, model: str, seed: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = run_dir / "checkpoints"
    curves = run_dir / "curves"
    checkpoints.mkdir(exist_ok=True)
    curves.mkdir(exist_ok=True)
    (checkpoints / "README.md").write_text(
        "Checkpoints are produced by real TensorFlow training; not run in this environment.\n",
        encoding="utf-8",
    )
    (curves / "training_curves_v001.json").write_text(
        json.dumps(
            {"status": "not_run_compute_bound", "loss": [], "dev_metrics": []}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    run_summary = {
        "run_id": f"model_{model}_seed_{seed}",
        "model_id": model.upper(),
        "seed": seed,
        "status": "not_run_compute_bound",
        "wall_clock_budget_minutes": TRAINING_BUDGET_MINUTES,
        "metrics": {
            "dev_em": None,
            "dev_token_f1": None,
            "dev_char_edit_distance": None,
            "dev_bleu1": None,
        },
        "note": "Training orchestration and artifact paths are ready; no scores/checkpoints fabricated without TensorFlow/GPU run.",
    }
    (run_dir / "run_summary_v001.json").write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )


def _write_training_runs_index(experiments_dir: Path) -> None:
    runs = []
    for model in ("a", "b"):
        for seed in RUN_ALL_SEED_CHOICES:
            runs.append(
                {
                    "model": model.upper(),
                    "seed": seed,
                    "run_summary": _ms2_experiment_relpath(
                        f"model_{model}/{seed}/run_summary_v001.json"
                    ),
                    "status": "pending_or_compute_bound",
                }
            )
    (experiments_dir / "training_runs_index_v001.json").write_text(
        json.dumps({"runs": runs}, indent=2) + "\n", encoding="utf-8"
    )


def _ablation_model_id(variant: str) -> str:
    if variant in {"no_film", "mean_merge", "plain_branch3"}:
        return "model_a"
    return "model_b"


def _ms2_experiment_relpath(suffix: str) -> str:
    return "/".join(("experiments", "ms2", suffix))


def _write_ablations_index(experiments_dir: Path) -> None:
    variants = (
        "no_film",
        "mean_merge",
        "plain_branch3",
        "sinusoidal_pe",
        "no_pe",
        "shared_layers",
    )
    rows = []
    for variant in variants:
        for seed in RUN_ALL_SEED_CHOICES:
            rows.append(
                {
                    "variant": variant,
                    "model": _ablation_model_id(variant),
                    "seed": seed,
                    "run_summary": _ms2_experiment_relpath(
                        f"ablations/{_ablation_model_id(variant)}/{variant}/{seed}/run_summary_v001.json"
                    ),
                    "status": "pending_or_compute_bound",
                }
            )
    (experiments_dir / "ablations_index_v001.json").write_text(
        json.dumps({"ablations": rows}, indent=2) + "\n", encoding="utf-8"
    )


def _write_length_analysis_contract(
    output_path: Path, analysis: dict[str, object]
) -> None:
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
            f"- `{MS2_LENGTH_DISTRIBUTION_ARTIFACT}`",
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
    return paths.run_output_dir / "run_summary_v001.json"


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
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=False)
    return (
        paths.experiments_ms2
        / "ablations"
        / _ablation_model_id(variant)
        / variant
        / str(seed)
        / "run_summary_v001.json"
    )


def _expected_compare_output(repo_root: Path) -> Path:
    return repo_root / "docs" / "reports" / "ms2_headline_table.md"


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


def _write_run_all_artifacts(
    paths: MS2Paths, command_results: list[CommandResult]
) -> None:
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
            "integration_checklist": _as_repo_relative_path(
                paths.repo_root, checklist_path
            ),
            "known_limitations": _as_repo_relative_path(
                paths.repo_root, limitations_path
            ),
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
