"""Choose which MS2 model the website ships: train every candidate the same way,
score each on the dev split, and report.

MS2 produced four from-scratch generators:

- ``model_a``: ADR tri-encoder RNN with gated fusion + FiLM decoder (``rnn/model_a.py``);
- ``rnn_model``: config-driven RNN variant with the same ideas (``rnn/model.py``);
- ``model_b``: RoPE encoder-decoder Transformer (``transformer/model_b.py``);
- ``transformer``: vanilla sinusoidal-PE Transformer (``transformer/model.py``).

The original MS2 trainer reads ``data/external/ms2-cleaned-input/*.json``,
which is not in the repository, so this recipe trains from the MS1 processed
dataset (``ms1 run-all`` output). Every candidate sees the same cached
examples (the Model A cache, which also carries ``encoder_input_ids``), the
same batch size, seed, label smoothing and gradient clipping, and stops
early on dev loss. Transformers use the MS2 Noam schedule; RNNs use the MS2
cosine-with-warmup schedule. Selection is by dev token-F1 under greedy
decoding (ties broken by EM, then dev loss).

    uv pip install tensorflow-cpu
    uv run python -m src.cli.ms1 run-all
    uv run python -m src.cli.ms3 train-ms2-web --model all
    uv run python -m src.cli.ms3 select-ms2-web
    uv run python -m src.cli.ms3 export-web
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.common.paths import resolve_ms2_paths

CANDIDATES = ("model_a", "rnn_model", "model_b", "transformer")
TRANSFORMERS = frozenset({"model_b", "transformer"})
CACHE_KEYS = (
    "encoder_input_ids",
    "decoder_input_ids",
    "decoder_target_ids",
    "loss_mask",
    "question_ids",
    "context_ids",
    "char_matrix",
)
CHECKPOINT_NAME = "web_best.weights.h5"
SUMMARY_NAME = "web_training_summary_v001.json"
SELECTION_JSON = Path("experiments/ms2/ms3_web_model_selection_v001.json")
SELECTION_REPORT = Path("docs/reports/ms3_web_model_selection.md")
VOCAB_SIZE = 4096
# Longest dev reference is 20 tokens (incl. EOS); 24 leaves headroom.
EVAL_MAX_DECODE = 24
EVAL_NAME = "web_dev_eval_v001.json"
DESCRIPTIONS = {
    "model_a": "Tri-encoder RNN, gated merge + FiLM-LSTM decoder (ADR §2)",
    "rnn_model": "Config-driven RNN variant (branches + FiLM decoder stack)",
    "model_b": "Encoder-decoder Transformer with RoPE (ADR §3)",
    "transformer": "Vanilla encoder-decoder Transformer, sinusoidal PE",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def prepare_cache(repo_root: Path) -> None:
    """Train/load the MS2 tokenizer and write the shared train/dev caches."""
    from src.ms2.data.pipeline import write_pipeline_cache
    from src.ms2.data.records import load_ms1_processed_records
    from src.ms2.data.tokenizer import load_tokenizer, train_tokenizer_assets

    records = load_ms1_processed_records(
        repo_root / "data/processed/ms1/ms1_dataset_processed_v001.jsonl"
    )
    if not records:
        raise FileNotFoundError("MS1 processed dataset missing; run `python -m src.cli.ms1 run-all`")
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    if paths.tokenizer_path.exists() and paths.char_vocab_path.exists():
        tokenizer = load_tokenizer(repo_root=repo_root)
    else:
        tokenizer = train_tokenizer_assets(records, repo_root=repo_root)
    for split in ("train", "dev"):
        cache = _cache_path(repo_root, split)
        if not cache.exists() or cache.stat().st_size == 0:
            write_pipeline_cache(records, tokenizer, repo_root=repo_root, target_model="a", split=split)


def _cache_path(repo_root: Path, split: str) -> Path:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True, split=f"{split}_a")
    return paths.tfrecord_shard_dir / "examples.jsonl"


def load_cache(repo_root: Path, split: str) -> dict[str, np.ndarray]:
    rows = [
        json.loads(line)
        for line in _cache_path(repo_root, split).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        key: np.array([row[key] for row in rows], dtype=np.float32 if key == "loss_mask" else np.int32)
        for key in CACHE_KEYS
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _char_vocab_size(repo_root: Path) -> int:
    path = resolve_ms2_paths(repo_root=repo_root).char_vocab_path
    return len(json.loads(path.read_text(encoding="utf-8")))


def build_candidate(name: str, repo_root: Path) -> tuple[Any, Callable[..., Any]]:
    """Return ``(model, forward)`` where ``forward(model, batch, training) -> logits``."""
    if name == "model_a":
        from src.ms2.models.rnn.model_a import ModelA

        model = ModelA(char_vocab_size=_char_vocab_size(repo_root))

        def forward(m: Any, b: dict, training: bool) -> Any:
            return m(
                (b["question_ids"], b["context_ids"], b["encoder_input_ids"], b["char_matrix"], b["decoder_input_ids"]),
                training=training,
            )["logits"]

    elif name == "rnn_model":
        from src.ms2.models.rnn.embedder import Embedder
        from src.ms2.models.rnn.model import RNNModel

        model = RNNModel(embedder=Embedder(vocab_size=VOCAB_SIZE))

        def forward(m: Any, b: dict, training: bool) -> Any:
            return m(
                b["question_ids"], b["context_ids"], b["encoder_input_ids"], b["decoder_input_ids"], training=training
            )

    elif name == "model_b":
        from src.ms2.models.transformer.model_b import ModelB

        model = ModelB()

        def forward(m: Any, b: dict, training: bool) -> Any:
            return m((b["encoder_input_ids"], b["decoder_input_ids"]), training=training)["logits"]

    elif name == "transformer":
        from src.ms2.models.transformer.model import TransformerModel

        model = TransformerModel(vocab_size=VOCAB_SIZE)

        def forward(m: Any, b: dict, training: bool) -> Any:
            return m(b["encoder_input_ids"], b["decoder_input_ids"], training=training)

    else:
        raise ValueError(f"unknown candidate {name!r}; choose from {CANDIDATES}")
    return model, forward


def _schedule(name: str, max_steps: int) -> Any:
    from src.ms2.training.schedules import CosineWithWarmup, Noam

    if name in TRANSFORMERS:
        return Noam(d_model=192, warmup_steps=800)
    return CosineWithWarmup(total_steps=max_steps, peak_lr=1e-3, min_lr=1e-5)


def run_dir(repo_root: Path, name: str, seed: int) -> Path:
    return resolve_ms2_paths(repo_root=repo_root, create_dirs=True, model=name, seed=seed).run_output_dir


def checkpoint_path(repo_root: Path, name: str, seed: int = 42) -> Path:
    return run_dir(repo_root, name, seed) / "checkpoints" / CHECKPOINT_NAME


def load_trained(name: str, repo_root: Path, seed: int = 42) -> tuple[Any, Callable[..., Any]]:
    import tensorflow as tf

    model, forward = build_candidate(name, repo_root)
    dev = load_cache(repo_root, "dev")
    forward(model, {k: tf.constant(v[:1]) for k, v in dev.items()}, False)
    model.load_weights(str(checkpoint_path(repo_root, name, seed)))
    return model, forward


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_candidate(
    repo_root: Path,
    name: str,
    budget_minutes: float = 40.0,
    max_steps: int = 8000,
    batch_size: int = 16,
    seed: int = 42,
    eval_every: int = 500,
    patience: int = 2,
) -> Path:
    """Train one candidate (label smoothing 0.1, clip 1.0); keep the best dev loss.

    The MS1 dataset is small (3.5k pairs over 13 transcripts), so the models
    overfit quickly: training stops after ``patience`` evaluations without a
    dev-loss improvement.
    """
    import tensorflow as tf

    from src.ms2.training.loss import label_smoothed_cross_entropy

    repo_root = repo_root.resolve()
    prepare_cache(repo_root)
    train, dev = load_cache(repo_root, "train"), load_cache(repo_root, "dev")
    print(f"[train-ms2-web] model={name} train={len(train['encoder_input_ids'])} dev={len(dev['encoder_input_ids'])}", flush=True)

    tf.keras.utils.set_random_seed(seed)
    dataset = tf.data.Dataset.from_tensor_slices(train).shuffle(4096, seed=seed).repeat().batch(batch_size).prefetch(2)
    model, forward = build_candidate(name, repo_root)
    optimizer = tf.keras.optimizers.Adam(
        _schedule(name, max_steps), beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipnorm=1.0
    )

    def loss_fn(batch: dict, training: bool, smoothing: float) -> Any:
        return label_smoothed_cross_entropy(
            forward(model, batch, training),
            batch["decoder_target_ids"],
            tf.cast(batch["loss_mask"], tf.bool),
            smoothing=smoothing,
        )

    @tf.function(reduce_retracing=True)
    def train_step(batch: dict) -> Any:
        with tf.GradientTape() as tape:
            loss = loss_fn(batch, training=True, smoothing=0.1)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            [(g, v) for g, v in zip(grads, model.trainable_variables, strict=False) if g is not None]
        )
        return loss

    @tf.function(reduce_retracing=True)
    def eval_step(batch: dict) -> Any:
        return loss_fn(batch, training=False, smoothing=0.0)

    def dev_loss() -> float:
        return float(np.mean([float(eval_step(b)) for b in tf.data.Dataset.from_tensor_slices(dev).batch(32)]))

    out_dir = run_dir(repo_root, name, seed)
    best_path = out_dir / "checkpoints" / CHECKPOINT_NAME
    best_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best, best_step, stale = float("inf"), 0, 0
    started = time.monotonic()
    step = 0
    for batch in dataset:
        step += 1
        loss = float(train_step(batch))
        if step % 100 == 0:
            print(f"[train-ms2-web] model={name} step={step} loss={loss:.4f} elapsed={time.monotonic() - started:.0f}s", flush=True)
        stop = time.monotonic() - started > budget_minutes * 60 or step >= max_steps
        if step % eval_every == 0 or stop:
            current = dev_loss()
            history.append({"step": step, "train_loss": loss, "dev_loss": current})
            print(f"[train-ms2-web] model={name} step={step} dev_loss={current:.4f}", flush=True)
            if current < best:
                best, best_step, stale = current, step, 0
                model.save_weights(str(best_path))
            else:
                stale += 1
        if stop or stale >= patience:
            break

    summary = {
        "model": name,
        "seed": seed,
        "batch_size": batch_size,
        "steps": step,
        "best_step": best_step,
        "best_dev_loss": best,
        "parameter_count": int(model.count_params()),
        "wall_clock_minutes": round((time.monotonic() - started) / 60.0, 2),
        "history": history,
    }
    (out_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return best_path


# ---------------------------------------------------------------------------
# Evaluation + selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateScore:
    model: str
    description: str
    parameter_count: int
    best_step: int
    best_dev_loss: float
    dev_em: float
    dev_token_f1: float
    dev_char_edit_distance: float
    dev_bleu1: float
    train_minutes: float


def greedy_decode_batch(model: Any, forward: Callable[..., Any], batch: dict, max_len: int) -> np.ndarray:
    """Full-recompute greedy decoding; valid for all four (causal) decoders."""
    import tensorflow as tf

    from src.ms2.data.tokenizer import BOS_ID, EOS_ID

    n, l_dec = batch["decoder_input_ids"].shape
    dec = np.zeros((n, l_dec), dtype=np.int32)
    dec[:, 0] = BOS_ID
    out = np.zeros((n, max_len), dtype=np.int32)
    done = np.zeros(n, dtype=bool)
    inputs = {k: tf.constant(v) for k, v in batch.items() if k != "decoder_input_ids"}

    @tf.function(reduce_retracing=True)
    def step_argmax(feed: dict, t: Any) -> Any:
        return tf.argmax(forward(model, feed, False)[:, t, :], axis=-1, output_type=tf.int32)

    for t in range(min(max_len, l_dec)):
        nxt = step_argmax({**inputs, "decoder_input_ids": tf.constant(dec)}, tf.constant(t)).numpy()
        nxt[done] = 0
        out[:, t] = nxt
        done |= nxt == EOS_ID
        if t + 1 < l_dec:
            dec[:, t + 1] = np.where(done, 0, nxt)
        if done.all():
            break
    return out


def evaluate_candidate(repo_root: Path, name: str, seed: int = 42, batch_size: int = 64) -> CandidateScore:
    from src.ms2.data.tokenizer import EOS_ID, PAD_ID, load_tokenizer
    from src.ms2.metrics.scoring import bleu1, char_edit_distance_normalized, exact_match, token_f1

    repo_root = repo_root.resolve()
    cached = run_dir(repo_root, name, seed) / EVAL_NAME
    ckpt = checkpoint_path(repo_root, name, seed)
    if cached.exists() and cached.stat().st_mtime > ckpt.stat().st_mtime:
        return CandidateScore(**json.loads(cached.read_text(encoding="utf-8")))
    tokenizer = load_tokenizer(repo_root=repo_root)
    model, forward = load_trained(name, repo_root, seed)
    dev = load_cache(repo_root, "dev")

    def text(ids: np.ndarray) -> str:
        kept: list[int] = []
        for token in ids:
            if int(token) in (EOS_ID, PAD_ID):
                break
            kept.append(int(token))
        return tokenizer.decode(kept)

    preds: list[str] = []
    for start in range(0, len(dev["encoder_input_ids"]), batch_size):
        batch = {k: v[start : start + batch_size] for k, v in dev.items()}
        preds.extend(text(row) for row in greedy_decode_batch(model, forward, batch, EVAL_MAX_DECODE))
    refs = [text(row) for row in dev["decoder_target_ids"]]

    summary = json.loads((run_dir(repo_root, name, seed) / SUMMARY_NAME).read_text(encoding="utf-8"))
    score = CandidateScore(
        model=name,
        description=DESCRIPTIONS[name],
        parameter_count=summary["parameter_count"],
        best_step=summary["best_step"],
        best_dev_loss=round(summary["best_dev_loss"], 4),
        dev_em=round(float(np.mean([exact_match(p, r) for p, r in zip(preds, refs, strict=True)])), 4),
        dev_token_f1=round(float(np.mean([token_f1(p, r) for p, r in zip(preds, refs, strict=True)])), 4),
        dev_char_edit_distance=round(
            float(np.mean([char_edit_distance_normalized(p, r) for p, r in zip(preds, refs, strict=True)])), 4
        ),
        dev_bleu1=round(float(np.mean([bleu1(p, r) for p, r in zip(preds, refs, strict=True)])), 4),
        train_minutes=summary["wall_clock_minutes"],
    )
    cached.write_text(json.dumps(asdict(score), indent=2) + "\n", encoding="utf-8")
    print(f"[select-ms2-web] evaluated model={name} dev_token_f1={score.dev_token_f1} dev_em={score.dev_em}", flush=True)
    return score


def rank(scores: list[CandidateScore]) -> list[CandidateScore]:
    return sorted(scores, key=lambda s: (-s.dev_token_f1, -s.dev_em, s.best_dev_loss))


def select_model(repo_root: Path, seed: int = 42) -> list[CandidateScore]:
    """Evaluate every trained candidate and write the selection JSON + report."""
    repo_root = repo_root.resolve()
    scores = [
        evaluate_candidate(repo_root, name, seed)
        for name in CANDIDATES
        if checkpoint_path(repo_root, name, seed).exists()
    ]
    if not scores:
        raise FileNotFoundError("no trained candidates; run `python -m src.cli.ms3 train-ms2-web --model all`")
    ranked = rank(scores)
    payload = {"selected": ranked[0].model, "metric": "dev_token_f1", "seed": seed, "candidates": [asdict(s) for s in ranked]}
    (repo_root / SELECTION_JSON).parent.mkdir(parents=True, exist_ok=True)
    (repo_root / SELECTION_JSON).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    n_dev = len(load_cache(repo_root, "dev")["encoder_input_ids"])
    (repo_root / SELECTION_REPORT).write_text(render_report(ranked, n_dev), encoding="utf-8")
    return ranked


def render_report(ranked: list[CandidateScore], n_dev: int = 414) -> str:
    rows = "\n".join(
        f"| {i} | `{s.model}` | {s.description} | {s.parameter_count:,} | {s.best_step} | {s.best_dev_loss:.3f} "
        f"| {s.dev_em:.3f} | {s.dev_token_f1:.3f} | {s.dev_char_edit_distance:.3f} | {s.dev_bleu1:.3f} | {s.train_minutes:.1f} |"
        for i, s in enumerate(ranked, start=1)
    )
    return f"""# MS3 Website: MS2 Model Selection

Generated by `python -m src.cli.ms3 select-ms2-web` (`src/ms3/models/ms2_web_selection.py`).

All four MS2 generators were trained the same way: same MS1-derived train
and dev caches, batch 16, seed 42, label smoothing 0.1, gradient clipping 1.0,
and early stopping after 2 dev-loss evaluations without improvement. Each was
then scored on the full dev split with greedy decoding. Candidates are ranked
by dev token-F1, then EM, then dev loss.

| Rank | Model | Architecture | Params | Best step | Dev loss | EM | Token-F1 | Char edit dist. ↓ | BLEU-1 | Train min |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

**Selected for the website:** `{ranked[0].model}`. It is the default MS2 model
on the site. The other candidates that have a NumPy web runtime (`model_a`,
`model_b`, `transformer`) stay selectable in the UI; `rnn_model` has no web
runtime.

Notes:

- The dev split has {n_dev} questions, so gaps of about 0.01 token-F1 are small.
  Read the ranking together with the other metrics, not as a decisive win.
- Dev loss and answer quality disagree: the lowest dev loss does not imply the
  best greedy answers. That is why selection uses decoded-answer metrics.
"""
