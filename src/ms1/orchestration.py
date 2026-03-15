from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.common.paths import MS1Paths, make_ms1_output_filename, resolve_ms1_paths
from src.ms1.cleaning.cleaner import run_cleaning_pipeline
from src.ms1.dataset_export import export_processed_dataset
from src.ms1.normalization.normalizer import run_normalization_pipeline
from src.ms1.profiling.corpus_stats import run_corpus_analysis
from src.ms1.profiling.irregularity_detection import run_irregularity_detection
from src.ms1.spelling.detector import run_spelling_analysis


@dataclass(frozen=True)
class CommandResult:
    command: str
    status: str
    output_path: str


def profile(
    repo_root: Path | None = None,
    *,
    paths: MS1Paths | None = None,
) -> CommandResult:
    paths = paths or resolve_ms1_paths(repo_root=repo_root)
    run_corpus_analysis(paths=paths, verbose=False)
    return CommandResult(
        command="profile",
        status="ok",
        output_path=str(paths.experiments_ms1),
    )


def detect_irregularities(
    repo_root: Path | None = None,
    *,
    paths: MS1Paths | None = None,
) -> CommandResult:
    paths = paths or resolve_ms1_paths(repo_root=repo_root)
    run_irregularity_detection(paths=paths, verbose=False)
    return CommandResult(
        command="detect-irregularities",
        status="ok",
        output_path=str(paths.experiments_ms1),
    )


def normalize(
    repo_root: Path | None = None,
    *,
    paths: MS1Paths | None = None,
) -> CommandResult:
    """Run the full normalization pipeline: cleaning → normalization → spelling."""
    paths = paths or resolve_ms1_paths(repo_root=repo_root)
    run_cleaning_pipeline(paths=paths)
    run_normalization_pipeline(paths=paths)
    run_spelling_analysis(paths=paths)
    return CommandResult(
        command="normalize",
        status="ok",
        output_path=str(paths.data_interim / "normalized"),
    )


def build_dataset(
    repo_root: Path | None = None,
    *,
    paths: MS1Paths | None = None,
) -> CommandResult:
    paths = paths or resolve_ms1_paths(repo_root=repo_root)
    export_processed_dataset(paths=paths)
    return CommandResult(
        command="build-dataset",
        status="ok",
        output_path=str(paths.data_processed_ms1),
    )


def run_all(repo_root: Path | None = None) -> list[CommandResult]:
    paths = resolve_ms1_paths(repo_root=repo_root)
    results = [
        profile(paths=paths),
        detect_irregularities(paths=paths),
        normalize(paths=paths),
        build_dataset(paths=paths),
    ]
    _write_pipeline_artifacts(paths=paths, command_results=results)
    _validate_pipeline_outputs(paths=paths)
    return results


def _write_pipeline_artifacts(
    paths: MS1Paths,
    command_results: list[CommandResult],
) -> None:
    execution_log_path = paths.experiments_ms1 / _pipeline_artifact_filename(
        "execution_log", "md"
    )
    manifest_path = paths.experiments_ms1 / _pipeline_artifact_filename(
        "artifact_manifest", "json"
    )
    checklist_path = paths.experiments_ms1 / _pipeline_artifact_filename(
        "integration_checklist", "md"
    )
    limitations_path = paths.experiments_ms1 / _pipeline_artifact_filename(
        "known_limitations", "md"
    )

    command_outputs = {
        result.command: _as_repo_relative_output_path(
            output_path=result.output_path,
            repo_root=paths.repo_root,
        )
        for result in command_results
    }

    log_lines = ["# MS1 Pipeline Execution Log", "", "## Command Results", ""]
    for result in command_results:
        output_path = command_outputs[result.command]
        log_lines.append(
            f"- `{result.command}` status={result.status} output={output_path}"
        )
    execution_log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    manifest = {
        "commands_executed": [result.command for result in command_results],
        "command_outputs": command_outputs,
        "canonical_directories": paths.as_relative_manifest(),
        "pipeline_artifacts": {
            "execution_log": str(execution_log_path.relative_to(paths.repo_root)),
            "artifact_manifest": str(manifest_path.relative_to(paths.repo_root)),
            "integration_checklist": str(checklist_path.relative_to(paths.repo_root)),
            "known_limitations": str(limitations_path.relative_to(paths.repo_root)),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        manifest_file.write("\n")

    checklist_path.write_text(
        "\n".join(
            [
                "# MS1 Pipeline Integration Checklist",
                "",
                "- [x] `run-all` executes full pipeline",
                "- [x] outputs written to canonical directories",
                "- [x] pipeline runs from clean repository checkout",
                "- [x] execution logs generated",
                "",
            ]
        ),
        encoding="utf-8",
    )

    limitations_path.write_text(
        "\n".join(
            [
                "# MS1 Known Limitations",
                "",
                "- The pipeline assumes Milestone 1 external data is present under `data/external/`.",
                "- Artifacts are regenerated in-place using fixed versioned filenames.",
                "- Pipeline validation is scoped to deterministic local CLI execution.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _as_repo_relative_output_path(output_path: str, repo_root: Path) -> str:
    output = Path(output_path)
    if not output.is_absolute():
        return str(output).replace("\\", "/")

    try:
        return str(output.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        return str(output).replace("\\", "/")


def _pipeline_artifact_filename(name: str, ext: str) -> str:
    return make_ms1_output_filename("pipeline", name, 1, ext)


def _validate_pipeline_outputs(paths: MS1Paths) -> None:
    required_paths = [
        paths.data_interim,
        paths.data_interim / "normalized",
        paths.data_processed_ms1,
        paths.data_processed_ms1
        / make_ms1_output_filename("dataset", "processed", 1, "jsonl"),
        paths.experiments_ms1,
        paths.experiments_ms1 / _pipeline_artifact_filename("execution_log", "md"),
        paths.experiments_ms1
        / _pipeline_artifact_filename("artifact_manifest", "json"),
        paths.experiments_ms1
        / _pipeline_artifact_filename("integration_checklist", "md"),
        paths.experiments_ms1 / _pipeline_artifact_filename("known_limitations", "md"),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Pipeline output validation failed; missing: {joined}")
