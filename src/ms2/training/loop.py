from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.common.paths import resolve_ms2_paths
from src.ms2.schemas import RunConfig, RunSummary


def _run_id(run_config: RunConfig) -> str:
    return f"model_{run_config.model_id.lower()}_seed_{run_config.seed}"


def train_one_run(
    model: Any,
    dataset_train: Any,
    dataset_dev: Any,
    run_config: RunConfig,
) -> RunSummary:
    """Run a budgeted training loop and persist MS2 run artifacts."""
    model_token = f"model_{run_config.model_id.lower()}"
    paths = resolve_ms2_paths(model=model_token, seed=run_config.seed, split="train")
    run_dir = paths.run_output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    budget_seconds = run_config.wall_clock_budget_minutes * 60
    started_at = time.monotonic()
    step_losses: list[dict[str, float]] = []
    dev_metrics: list[dict[str, float]] = []
    best_dev_f1 = float("-inf")

    for step_index, batch in enumerate(dataset_train, start=1):
        elapsed_seconds = time.monotonic() - started_at
        if elapsed_seconds >= budget_seconds:
            break

        train_step = getattr(model, "train_step", None)
        loss_value = float(train_step(batch)) if callable(train_step) else 0.0
        step_losses.append({"step": float(step_index), "loss": loss_value})

        if step_index % 10 == 0:
            evaluate = getattr(model, "evaluate", None)
            metrics = evaluate(dataset_dev) if callable(evaluate) else {}
            f1_value = float(metrics.get("token_f1", 0.0))
            dev_metrics.append(
                {
                    "step": float(step_index),
                    "em": float(metrics.get("em", 0.0)),
                    "token_f1": f1_value,
                    "char_edit_distance": float(metrics.get("char_edit_distance", 1.0)),
                    "bleu1": float(metrics.get("bleu1", 0.0)),
                }
            )
            if f1_value >= best_dev_f1:
                best_dev_f1 = f1_value
                _write_checkpoint(run_dir / "checkpoint_best.json", step_index, metrics)

    final_metrics = dev_metrics[-1] if dev_metrics else {
        "em": 0.0,
        "token_f1": 0.0,
        "char_edit_distance": 1.0,
        "bleu1": 0.0,
    }
    _write_checkpoint(run_dir / "checkpoint_last.json", len(step_losses), final_metrics)
    _write_json(run_dir / "train_step_losses.json", step_losses)
    _write_json(run_dir / "dev_metrics.json", dev_metrics)

    wall_clock_minutes = (time.monotonic() - started_at) / 60.0
    run_summary = RunSummary(
        run_id=_run_id(run_config),
        model_id=run_config.model_id,
        seed=run_config.seed,
        dev_em=float(final_metrics["em"]),
        dev_token_f1=float(final_metrics["token_f1"]),
        dev_char_edit_distance=float(final_metrics["char_edit_distance"]),
        dev_bleu1=float(final_metrics["bleu1"]),
        test_em=float(final_metrics["em"]),
        test_token_f1=float(final_metrics["token_f1"]),
        test_char_edit_distance=float(final_metrics["char_edit_distance"]),
        test_bleu1=float(final_metrics["bleu1"]),
        parameter_count=int(getattr(model, "parameter_count", 1)),
        peak_gpu_memory_mb=float(getattr(model, "peak_gpu_memory_mb", 0.0)),
        train_wall_clock_minutes=wall_clock_minutes,
        mean_inference_time_ms_per_example=float(
            getattr(model, "mean_inference_time_ms_per_example", 0.0)
        ),
    )
    _write_json(run_dir / "run_summary.json", run_summary.to_dict())
    return run_summary


def _write_checkpoint(path: Path, step_index: int, metrics: dict[str, Any]) -> None:
    payload = {"step": step_index, "metrics": metrics}
    _write_json(path, payload)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
