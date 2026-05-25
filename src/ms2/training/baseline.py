from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

from tqdm.auto import tqdm

from src.common.paths import resolve_ms2_paths
from src.ms2.data.length_caps import LENGTH_CAPS
from src.ms2.data.pipeline import write_pipeline_cache
from src.ms2.data.records import load_ms2_cleaned_input_records
from src.ms2.data.tokenizer import ensure_default_tokenizer
from src.ms2.schemas import RunConfig, RunSummary
from src.ms2.training.loss import label_smoothed_cross_entropy

ModelToken = Literal["a", "b"]


def train_baseline_run(
    repo_root: Path,
    model_token: ModelToken,
    seed: int,
    run_config: RunConfig | None = None,
) -> RunSummary:
    """Train one MS2 baseline run with real TensorFlow optimization."""
    config = run_config or _default_run_config(model_token, seed)
    paths = resolve_ms2_paths(
        repo_root=repo_root, create_dirs=True, model=f"model_{model_token}", seed=seed
    )

    train_cache_path = (
        resolve_ms2_paths(
            repo_root=repo_root, create_dirs=True, split=f"train_{model_token}"
        ).tfrecord_shard_dir
        / "examples.jsonl"
    )
    dev_cache_path = (
        resolve_ms2_paths(
            repo_root=repo_root, create_dirs=True, split=f"dev_{model_token}"
        ).tfrecord_shard_dir
        / "examples.jsonl"
    )

    if not _cache_has_examples(train_cache_path) or not _cache_has_examples(
        dev_cache_path
    ):
        records = load_ms2_cleaned_input_records(
            resolve_ms2_paths(repo_root=repo_root, create_dirs=False)
        )
        tokenizer = ensure_default_tokenizer(repo_root=repo_root)
        write_pipeline_cache(
            records,
            tokenizer,
            repo_root=repo_root,
            target_model=model_token,
            split="train",
        )
        write_pipeline_cache(
            records,
            tokenizer,
            repo_root=repo_root,
            target_model=model_token,
            split="dev",
        )

    tf = _tf()
    tf.keras.utils.set_random_seed(seed)
    train_ds = _dataset_from_cache(
        train_cache_path,
        model_token=model_token,
        batch_size=_batch_size(config),
        shuffle=True,
    ).repeat()
    dev_ds = _dataset_from_cache(
        dev_cache_path,
        model_token=model_token,
        batch_size=_batch_size(config),
        shuffle=False,
    )
    model = _build_model(model_token)
    optimizer = tf.keras.optimizers.Adam(_schedule(model_token, config))
    run_dir = paths.run_output_dir
    curves_dir = run_dir / "curves"
    checkpoints_dir = run_dir / "checkpoints"
    curves_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    budget_seconds = max(1.0, config.wall_clock_budget_minutes * 60.0)
    max_steps = _max_steps_for_budget(config)
    log_every = _log_every_steps(default=10)
    _print_training_banner(
        model_token=model_token,
        seed=seed,
        batch_size=_batch_size(config),
        budget_seconds=budget_seconds,
        max_steps=max_steps,
    )
    losses: list[dict[str, float]] = []
    best_loss = float("inf")
    iterator = iter(train_ds)
    progress = tqdm(total=max_steps, desc="train", unit="step", dynamic_ncols=True)
    for step in range(1, max_steps + 1):
        elapsed_now = time.monotonic() - started
        if elapsed_now >= budget_seconds:
            tqdm.write(
                "[train] budget reached "
                f"elapsed_seconds={elapsed_now:.1f} "
                f"step={step - 1}/{max_steps}"
            )
            break
        batch = next(iterator)
        with tf.GradientTape() as tape:
            logits = _forward_logits(model, model_token, batch, training=True)
            loss = label_smoothed_cross_entropy(
                logits,
                batch["decoder_target_ids"],
                batch["loss_mask"],
                smoothing=config.label_smoothing,
            )
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            (grad, var)
            for grad, var in zip(grads, model.trainable_variables, strict=False)
            if grad is not None
        )
        loss_value = float(loss.numpy())
        losses.append({"step": step, "loss": loss_value})
        progress.update(1)
        progress.set_postfix(loss=f"{loss_value:.4f}", refresh=False)
        if step == 1 or step % log_every == 0 or step == max_steps:
            elapsed = max(1e-6, time.monotonic() - started)
            steps_per_sec = step / elapsed
            tqdm.write(
                "[train] progress "
                f"step={step}/{max_steps} "
                f"loss={loss_value:.6f} "
                f"elapsed_seconds={elapsed:.1f} "
                f"steps_per_sec={steps_per_sec:.3f}"
            )
        if loss_value < best_loss:
            best_loss = loss_value
            (checkpoints_dir / "best.weights.h5").parent.mkdir(
                parents=True, exist_ok=True
            )
            model.save_weights(str(checkpoints_dir / "best.weights.h5"))
            tqdm.write(
                "[train] checkpoint best_updated "
                f"step={step} "
                f"loss={loss_value:.6f} "
                f"path={checkpoints_dir / 'best.weights.h5'}"
            )
    progress.close()
    model.save_weights(str(checkpoints_dir / "last.weights.h5"))
    dev_loss = _evaluate_loss(model, model_token, dev_ds, config)
    wall_clock = (time.monotonic() - started) / 60.0
    (curves_dir / "train_step_losses.json").write_text(
        json.dumps(losses, indent=2) + "\n", encoding="utf-8"
    )
    (curves_dir / "dev_metrics.json").write_text(
        json.dumps([{"step": len(losses), "loss": dev_loss}], indent=2) + "\n",
        encoding="utf-8",
    )
    summary = RunSummary(
        run_id=f"model_{model_token}_seed_{seed}",
        model_id=model_token.upper(),
        seed=seed,
        dev_em=0.0,
        dev_token_f1=0.0,
        dev_char_edit_distance=1.0,
        dev_bleu1=0.0,
        test_em=0.0,
        test_token_f1=0.0,
        test_char_edit_distance=1.0,
        test_bleu1=0.0,
        parameter_count=int(model.count_params()),
        peak_gpu_memory_mb=_peak_gpu_memory_mb(),
        train_wall_clock_minutes=wall_clock,
        mean_inference_time_ms_per_example=0.0,
    )
    payload = summary.to_dict() | {
        "status": "trained",
        "train_steps": len(losses),
        "final_train_loss": losses[-1]["loss"] if losses else None,
        "best_train_loss": best_loss if losses else None,
        "dev_loss": dev_loss,
        "wall_clock_budget_minutes": config.wall_clock_budget_minutes,
    }
    (run_dir / "run_summary_v001.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "[train] completed "
        f"steps={len(losses)} "
        f"best_train_loss={best_loss if losses else 'n/a'} "
        f"dev_loss={dev_loss:.6f} "
        f"summary={run_dir / 'run_summary_v001.json'}",
        flush=True,
    )
    return summary


def _log_every_steps(default: int = 10) -> int:
    raw = os.environ.get("MS2_LOG_EVERY", str(default)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return max(1, parsed)


def _print_training_banner(
    model_token: ModelToken,
    seed: int,
    batch_size: int,
    budget_seconds: float,
    max_steps: int,
) -> None:
    tf = _tf()
    gpu_devices = tf.config.list_physical_devices("GPU")
    gpu_desc = (
        ", ".join(device.name for device in gpu_devices) if gpu_devices else "none"
    )
    print(
        "[train] start "
        f"model={model_token.upper()} "
        f"seed={seed} "
        f"batch_size={batch_size} "
        f"budget_seconds={budget_seconds:.1f} "
        f"max_steps={max_steps} "
        f"gpus={gpu_desc}",
        flush=True,
    )


def _default_run_config(model_token: ModelToken, seed: int) -> RunConfig:
    return RunConfig(
        model_id=model_token.upper(),
        seed=seed,
        lr_schedule="cosine_with_warmup" if model_token == "a" else "noam",
        wall_clock_budget_minutes=0.25,
        target_tokens_per_batch=420,
    )


def _batch_size(config: RunConfig) -> int:
    return max(1, min(8, config.target_tokens_per_batch // max(config.l_enc, 1)))


def _max_steps_for_budget(config: RunConfig) -> int:
    return max(1, int(config.wall_clock_budget_minutes * 60 // 2) or 1)


def _schedule(model_token: ModelToken, config: RunConfig) -> Any:
    from src.ms2.training.schedules import CosineWithWarmup, Noam

    if model_token == "a":
        return CosineWithWarmup(total_steps=max(10, _max_steps_for_budget(config)))
    return Noam(d_model=192, warmup_steps=1000)


def _build_model(model_token: ModelToken) -> Any:
    if model_token == "a":
        from src.ms2.models.rnn.model_a import ModelA

        return ModelA()
    from src.ms2.models.transformer.model_b import ModelB

    return ModelB()


def _dataset_from_cache(
    cache_path: Path, model_token: ModelToken, batch_size: int, shuffle: bool
) -> Any:
    tf = _tf()
    rows = [
        json.loads(line)
        for line in cache_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        rows = [_empty_row(model_token)]

    tensor_dict = {}
    for row in rows:
        processed_row = _tensor_row(row, model_token)
        for key, val in processed_row.items():
            if key not in tensor_dict:
                tensor_dict[key] = []
            tensor_dict[key].append(val)

    spec = _output_signature(model_token)
    for key, val_list in tensor_dict.items():
        tensor_dict[key] = tf.constant(val_list, dtype=spec[key].dtype)

    ds = tf.data.Dataset.from_tensor_slices(tensor_dict)
    if shuffle:
        ds = ds.shuffle(min(len(rows), 1024), seed=13, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _cache_has_examples(path: Path) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size == 0:
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return True
    return False


def _tensor_row(row: dict[str, Any], model_token: ModelToken) -> dict[str, Any]:
    data = {
        "encoder_input_ids": row["encoder_input_ids"],
        "decoder_input_ids": row["decoder_input_ids"],
        "decoder_target_ids": row["decoder_target_ids"],
        "loss_mask": row["loss_mask"],
    }
    if model_token == "a":
        q, c = _split_model_a_inputs(row["encoder_input_ids"])
        data["question_ids"] = q
        data["context_ids"] = c
        data["char_matrix"] = row["char_matrix"]
    return data


def _output_signature(model_token: ModelToken) -> dict[str, Any]:
    tf = _tf()
    spec = {
        "encoder_input_ids": tf.TensorSpec((LENGTH_CAPS["l_enc"],), tf.int32),
        "decoder_input_ids": tf.TensorSpec((LENGTH_CAPS["l_dec"],), tf.int32),
        "decoder_target_ids": tf.TensorSpec((LENGTH_CAPS["l_dec"],), tf.int32),
        "loss_mask": tf.TensorSpec((LENGTH_CAPS["l_dec"],), tf.bool),
    }
    if model_token == "a":
        spec |= {
            "question_ids": tf.TensorSpec((LENGTH_CAPS["l_q"],), tf.int32),
            "context_ids": tf.TensorSpec((LENGTH_CAPS["l_c"],), tf.int32),
            "char_matrix": tf.TensorSpec((LENGTH_CAPS["l_enc"], 16), tf.int32),
        }
    return spec


def _split_model_a_inputs(encoder_ids: list[int]) -> tuple[list[int], list[int]]:
    sep_index = encoder_ids.index(4) if 4 in encoder_ids else 1
    eos_index = encoder_ids.index(3) if 3 in encoder_ids else len(encoder_ids)
    question = encoder_ids[1:sep_index][: LENGTH_CAPS["l_q"]]
    context = encoder_ids[sep_index + 1 : eos_index][: LENGTH_CAPS["l_c"]]
    return _pad(question, LENGTH_CAPS["l_q"]), _pad(context, LENGTH_CAPS["l_c"])


def _pad(values: list[int], length: int) -> list[int]:
    return values[:length] + [0] * max(0, length - len(values))


def _forward_logits(
    model: Any, model_token: ModelToken, batch: dict[str, Any], training: bool
) -> Any:
    if model_token == "a":
        outputs = model(
            (
                batch["question_ids"],
                batch["context_ids"],
                batch["encoder_input_ids"],
                batch["char_matrix"],
                batch["decoder_input_ids"],
            ),
            training=training,
        )
    else:
        outputs = model(
            (batch["encoder_input_ids"], batch["decoder_input_ids"]), training=training
        )
    return outputs["logits"]


def _evaluate_loss(
    model: Any, model_token: ModelToken, dataset: Any, config: RunConfig
) -> float:
    losses = []
    for idx, batch in enumerate(dataset):
        if idx >= 10:
            break
        logits = _forward_logits(model, model_token, batch, training=False)
        loss = label_smoothed_cross_entropy(
            logits,
            batch["decoder_target_ids"],
            batch["loss_mask"],
            smoothing=config.label_smoothing,
        )
        losses.append(float(loss.numpy()))
    return sum(losses) / len(losses) if losses else 0.0


def _empty_row(model_token: ModelToken) -> dict[str, Any]:
    row = {
        "encoder_input_ids": [0] * LENGTH_CAPS["l_enc"],
        "decoder_input_ids": [0] * LENGTH_CAPS["l_dec"],
        "decoder_target_ids": [0] * LENGTH_CAPS["l_dec"],
        "loss_mask": [False] * LENGTH_CAPS["l_dec"],
    }
    if model_token == "a":
        row["char_matrix"] = [[0] * 16 for _ in range(LENGTH_CAPS["l_enc"])]
    return row


def _peak_gpu_memory_mb() -> float:
    tf = _tf()
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
    except Exception:
        return 0.0
    return float(info.get("peak", 0)) / (1024 * 1024)


def _tf() -> Any:
    import tensorflow as tf

    return tf
