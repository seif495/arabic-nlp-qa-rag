"""Train the MS2 Model B (RoPE Transformer) checkpoint shipped with the website.

The original MS2 trainer reads ``data/external/ms2-cleaned-input/*.json``,
which is not part of the repository. This recipe trains the same ``ModelB``
directly from the MS1 processed dataset (``ms1 run-all`` output) so the web
checkpoint can be reproduced from the repo alone:

    uv pip install tensorflow-cpu
    uv run python -m src.cli.ms1 run-all
    uv run python -m src.cli.ms3 train-model-b
    uv run python -m src.cli.ms3 export-web
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.common.paths import resolve_ms2_paths

CACHE_KEYS = ("encoder_input_ids", "decoder_input_ids", "decoder_target_ids", "loss_mask")


def _load_cache(repo_root: Path, split: str) -> dict[str, np.ndarray]:
    path = (
        resolve_ms2_paths(repo_root=repo_root, create_dirs=True, split=f"{split}_b").tfrecord_shard_dir
        / "examples.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        key: np.array([row[key] for row in rows], dtype=np.float32 if key == "loss_mask" else np.int32)
        for key in CACHE_KEYS
    }


def prepare_model_b_cache(repo_root: Path) -> None:
    """Train/load the MS2 tokenizer and write Model B train/dev caches from MS1 records."""
    from src.ms2.data.pipeline import write_pipeline_cache
    from src.ms2.data.records import load_ms1_processed_records
    from src.ms2.data.tokenizer import load_tokenizer, train_tokenizer_assets

    records = load_ms1_processed_records(repo_root / "data/processed/ms1/ms1_dataset_processed_v001.jsonl")
    if not records:
        raise FileNotFoundError("MS1 processed dataset missing; run `python -m src.cli.ms1 run-all`")
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    if paths.tokenizer_path.exists() and paths.char_vocab_path.exists():
        tokenizer = load_tokenizer(repo_root=repo_root)
    else:
        tokenizer = train_tokenizer_assets(records, repo_root=repo_root)
    for split in ("train", "dev"):
        write_pipeline_cache(records, tokenizer, repo_root=repo_root, target_model="b", split=split)


def train_model_b_for_web(
    repo_root: Path,
    budget_minutes: float = 60.0,
    batch_size: int = 16,
    seed: int = 42,
    eval_every: int = 500,
    patience: int = 2,
) -> Path:
    """Train ``ModelB`` (Noam/Adam, label smoothing 0.1, clip 1.0); keep best dev loss.

    The MS1 dataset is small (3.5k pairs over 13 transcripts), so Model B
    overfits quickly: training stops after ``patience`` evaluations without a
    dev-loss improvement.
    """
    import tensorflow as tf

    from src.ms2.models.transformer.model_b import ModelB
    from src.ms2.training.loss import label_smoothed_cross_entropy
    from src.ms2.training.schedules import Noam

    repo_root = repo_root.resolve()
    prepare_model_b_cache(repo_root)
    train, dev = _load_cache(repo_root, "train"), _load_cache(repo_root, "dev")
    print(f"[train-model-b] train={len(train['encoder_input_ids'])} dev={len(dev['encoder_input_ids'])}", flush=True)

    tf.keras.utils.set_random_seed(seed)
    dataset = (
        tf.data.Dataset.from_tensor_slices(train)
        .shuffle(4096, seed=seed)
        .repeat()
        .batch(batch_size)
        .prefetch(2)
    )
    model = ModelB()
    optimizer = tf.keras.optimizers.Adam(
        Noam(d_model=192, warmup_steps=800), beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipnorm=1.0
    )

    def loss_fn(batch: dict, training: bool, smoothing: float) -> tf.Tensor:
        logits = model((batch["encoder_input_ids"], batch["decoder_input_ids"]), training=training)["logits"]
        return label_smoothed_cross_entropy(
            logits, batch["decoder_target_ids"], tf.cast(batch["loss_mask"], tf.bool), smoothing=smoothing
        )

    @tf.function(reduce_retracing=True)
    def train_step(batch: dict) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = loss_fn(batch, training=True, smoothing=0.1)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            [(g, v) for g, v in zip(grads, model.trainable_variables, strict=False) if g is not None]
        )
        return loss

    @tf.function(reduce_retracing=True)
    def eval_step(batch: dict) -> tf.Tensor:
        return loss_fn(batch, training=False, smoothing=0.0)

    def dev_loss() -> float:
        return float(np.mean([float(eval_step(b)) for b in tf.data.Dataset.from_tensor_slices(dev).batch(32)]))

    out_dir = resolve_ms2_paths(repo_root=repo_root, create_dirs=True, model="model_b", seed=seed).run_output_dir
    checkpoints = out_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints / "best.weights.h5"
    history: list[dict[str, float]] = []
    best = float("inf")
    stale_evals = 0
    started = time.monotonic()
    step = 0
    for batch in dataset:
        step += 1
        loss = float(train_step(batch))
        if step % 50 == 0:
            elapsed = time.monotonic() - started
            print(f"[train-model-b] step={step} loss={loss:.4f} elapsed={elapsed:.0f}s", flush=True)
        out_of_budget = time.monotonic() - started > budget_minutes * 60
        if step % eval_every == 0 or out_of_budget:
            current = dev_loss()
            history.append({"step": step, "train_loss": loss, "dev_loss": current})
            print(f"[train-model-b] step={step} dev_loss={current:.4f}", flush=True)
            if current < best:
                best, stale_evals = current, 0
                model.save_weights(str(best_path))
            else:
                stale_evals += 1
        if out_of_budget or stale_evals >= patience:
            break

    (out_dir / "web_training_curve_v001.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "batch_size": batch_size,
                "steps": step,
                "best_dev_loss": best,
                "parameter_count": int(model.count_params()),
                "wall_clock_minutes": (time.monotonic() - started) / 60.0,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return best_path
