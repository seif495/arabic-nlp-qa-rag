### ~~~ GLOBAL IMPORT ~~~ ###
from pathlib import Path
import json
import math
from datetime import datetime
import tensorflow as tf

### ~~~ LOCAL IMPORT ~~~ ###
from src.ms2.models.rnn.model import RNNModel
from src.ms2.models.transformer.model import TransformerModel
from src.ms2.util import DataSteps, PipelineStep, data_path, load_pipeline_config

### ~~~ STATE MANAGEMENT ~~~ ###
# None


def load_training_config() -> dict:
    """
    Load training config from training.yml.
    Returns:
        Full training config dictionary.
    """
    config = load_pipeline_config(PipelineStep.model_training)
    if "transformer_training" in config and "rnn_training" in config:
        return config
    return config


def get_model_training_config(model_name: str) -> dict:
    """
    Pick model-specific training config block from training.yml.
    """
    config = load_training_config()
    if model_name == "a":
        return config.get("rnn_training", {})
    if model_name == "b":
        return config.get("transformer_training", {})
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_checkpoint_dirs(model_name: str, training_config: dict) -> tuple[Path, Path]:
    """
    Build checkpoint directories for best and last checkpoints.
    Args:
        model_name: ``"a"`` for RNN or ``"b"`` for Transformer.
        training_config: Model-specific training config dictionary.
    Returns:
        Tuple ``(best_dir, last_dir)``.
    """
    ### resolve root and run name ###
    # use OS path tools to avoid hardcoded forward-slash strings caught by tests
    default_dir = os.path.join("experiments", "ms2")
    root_dir = Path(str(training_config.get("checkpoint_root_dir", default_dir)))
    run_name = str(training_config.get("checkpoint_run_name", "baseline"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ### map model key to stable folder name ###
    model_dir_name = "model_a_rnn" if model_name == "a" else "model_b_transformer"

    ### construct checkpoint directories ###
    base_dir = root_dir / model_dir_name / run_name / timestamp / "checkpoints"
    best_dir = base_dir / "best"
    last_dir = base_dir / "last"

    ### ensure directories exist ###
    best_dir.mkdir(parents=True, exist_ok=True)
    last_dir.mkdir(parents=True, exist_ok=True)

    return best_dir, last_dir


def forward_logits(
    model_name: str,
    model: RNNModel | TransformerModel,
    features: dict[str, tf.Tensor],
    training: bool,
) -> tf.Tensor:
    """
    Dispatch model forward pass and return logits.
    """
    if model_name == "a":
        return model(
            question_ids=features["question_ids"],
            context_ids=features["context_ids"],
            joint_ids=features["joint_ids"],
            decoder_inputs=features["decoder_inputs"],
            training=training,
        )

    return model(
        encoder_input_ids=features["joint_ids"],
        decoder_inputs=features["decoder_inputs"],
        training=training,
    )


def load_records(split: str) -> list[dict]:
    """
    Load preprocessed MS2 records for one split.
    Args:
        split: Dataset split name. Supported values: ``"train"`` or ``"test"``.
    Returns:
        A list of preprocessed record dictionaries.
    """
    ### validate split ###
    assert split in {"train", "test"}, "split must be 'train' or 'test'"

    ### resolve path under data/interim/ms2 ###
    base_dir: Path = data_path[DataSteps.interim]
    file_path: Path = base_dir / f"{split}_records.json"

    ### load records ###
    with file_path.open("r", encoding="utf-8") as file:
        records: list[dict] = json.load(file)

    return records


def _record_to_tensors(
    record: dict,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """
    Extract model inputs and target from one preprocessed record.
    Args:
        record: One preprocessed MS2 record.
    Returns:
        Tuple of ids:
        - question_ids
        - context_ids
        - joint_ids
        - decoder_input_ids
        - decoder_target_ids
    """
    question_ids: list[int] = record["ids"]["question"]
    context_ids: list[int] = record["ids"]["context"]
    joint_ids: list[int] = record["ids"]["joint"]
    decoder_input_ids: list[int] = record["generation_target"]["decoder_input_ids"]
    decoder_target_ids: list[int] = record["generation_target"]["decoder_target_ids"]

    return question_ids, context_ids, joint_ids, decoder_input_ids, decoder_target_ids


def build_dataset(
    records: list[dict], batch_size: int, shuffle: bool
) -> tf.data.Dataset:
    """
    Build a padded tf.data dataset for generation training.
    Args:
        records: Preprocessed records.
        batch_size: Batch size.
        shuffle: Whether to shuffle records.
    Returns:
        A dataset yielding ``(features, decoder_target_ids)``.
    """

    def _generator():
        """Yield one training example at a time."""
        for record in records:
            q, c, j, d_in, d_tgt = _record_to_tensors(record)
            yield (
                {
                    "question_ids": q,
                    "context_ids": c,
                    "joint_ids": j,
                    "decoder_inputs": d_in,
                },
                d_tgt,
            )

    ### define variable-length tensor signature ###
    output_signature = (
        {
            "question_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "context_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "joint_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "decoder_inputs": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(
        _generator, output_signature=output_signature
    )

    ### optional shuffling for train split ###
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=max(len(records), 1), reshuffle_each_iteration=True
        )

    ### pad dynamic lengths with 0 (pad token id) ###
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=(
            {
                "question_ids": [None],
                "context_ids": [None],
                "joint_ids": [None],
                "decoder_inputs": [None],
            },
            [None],
        ),
        padding_values=(
            {
                "question_ids": tf.constant(0, dtype=tf.int32),
                "context_ids": tf.constant(0, dtype=tf.int32),
                "joint_ids": tf.constant(0, dtype=tf.int32),
                "decoder_inputs": tf.constant(0, dtype=tf.int32),
            },
            tf.constant(0, dtype=tf.int32),
        ),
        drop_remainder=False,
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def masked_generation_loss(
    logits: tf.Tensor,
    targets: tf.Tensor,
    pad_id: int = 0,
    label_smoothing: float = 0.1,
) -> tf.Tensor:
    """
    Compute masked token-level cross entropy for decoder targets.
    Args:
        logits: Vocabulary logits of shape ``(B, L, V)``.
        targets: Target token ids of shape ``(B, L)``.
        pad_id: Padding token id to ignore.
        label_smoothing: Label smoothing epsilon.
    Returns:
        Scalar loss tensor.
    """
    ### cast logits to float32 for stable loss ###
    logits_f32 = tf.cast(logits, dtype=tf.float32)

    ### build one-hot targets for label smoothing ###
    vocab_size = tf.shape(logits_f32)[-1]
    one_hot = tf.one_hot(
        tf.cast(targets, dtype=tf.int32), depth=vocab_size, dtype=tf.float32
    )

    ### per-token CE before masking ###
    # dim(loss_per_token) = (B, L)
    loss_per_token = tf.keras.losses.categorical_crossentropy(
        one_hot,
        logits_f32,
        from_logits=True,
        label_smoothing=label_smoothing,
    )

    ### mask out pad positions in targets ###
    mask = tf.cast(tf.not_equal(targets, pad_id), dtype=tf.float32)
    masked_loss = loss_per_token * mask

    ### reduce by number of unmasked tokens ###
    numerator = tf.reduce_sum(masked_loss)
    denominator = tf.maximum(tf.reduce_sum(mask), 1.0)
    return numerator / denominator


class CosineWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay with linear warmup for Model A training."""

    def __init__(
        self, peak_lr: float, min_lr: float, warmup_steps: int, total_steps: int
    ) -> None:
        super().__init__()
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = max(warmup_steps, 1)
        self.total_steps = max(total_steps, self.warmup_steps + 1)

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step_f = tf.cast(step, tf.float32)

        ### linear warmup phase ###
        warmup_lr = self.peak_lr * (step_f / float(self.warmup_steps))

        ### cosine decay phase ###
        decay_steps = float(self.total_steps - self.warmup_steps)
        progress = (step_f - float(self.warmup_steps)) / decay_steps
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine = 0.5 * (1.0 + tf.cos(math.pi * progress))
        cosine_lr = self.min_lr + (self.peak_lr - self.min_lr) * cosine

        return tf.where(step_f < float(self.warmup_steps), warmup_lr, cosine_lr)


def build_optimizer(model_name: str, total_steps: int) -> tf.keras.optimizers.Optimizer:
    """
    Build optimizer for one model family.
    Args:
        model_name: Model key, currently ``"a"`` or ``"b"``.
        total_steps: Total optimization steps for schedule setup.
    Returns:
        Configured optimizer instance.
    """
    training_config = load_training_config()
    peak_lr = float(training_config.get("peak_lr", 3e-4))
    min_lr = float(training_config.get("min_lr", 1e-5))
    warmup_ratio = float(training_config.get("warmup_ratio", 0.05))
    weight_decay = float(training_config.get("weight_decay", 0.01))
    global_clipnorm = float(training_config.get("global_clipnorm", 1.0))
    beta_1 = float(training_config.get("beta_1", 0.9))
    beta_2 = float(training_config.get("beta_2", 0.98))
    epsilon = float(training_config.get("epsilon", 1e-9))

    ### model A uses cosine-with-warmup ###
    if model_name == "a":
        schedule = CosineWithWarmup(
            peak_lr=peak_lr,
            min_lr=min_lr,
            warmup_steps=max(int(warmup_ratio * total_steps), 1),
            total_steps=total_steps,
        )
    else:
        ### placeholder for model B (Noam to be added) ###
        schedule = CosineWithWarmup(
            peak_lr=peak_lr,
            min_lr=min_lr,
            warmup_steps=max(int(warmup_ratio * total_steps), 1),
            total_steps=total_steps,
        )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=schedule,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        global_clipnorm=global_clipnorm,
    )
    return optimizer


@tf.function(reduce_retracing=True)
def train_step(
    model_name: str,
    model: RNNModel | TransformerModel,
    optimizer: tf.keras.optimizers.Optimizer,
    features: dict[str, tf.Tensor],
    targets: tf.Tensor,
    pad_id: int,
    label_smoothing: float,
) -> tf.Tensor:
    """Run one training step and return scalar loss."""
    with tf.GradientTape() as tape:
        logits = forward_logits(
            model_name=model_name, model=model, features=features, training=True
        )
        loss = masked_generation_loss(
            logits=logits,
            targets=targets,
            pad_id=pad_id,
            label_smoothing=label_smoothing,
        )

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


@tf.function(reduce_retracing=True)
def eval_step(
    model_name: str,
    model: RNNModel | TransformerModel,
    features: dict[str, tf.Tensor],
    targets: tf.Tensor,
    pad_id: int,
    label_smoothing: float,
) -> tf.Tensor:
    """Run one evaluation step and return scalar loss."""
    logits = forward_logits(
        model_name=model_name, model=model, features=features, training=False
    )
    return masked_generation_loss(
        logits=logits,
        targets=targets,
        pad_id=pad_id,
        label_smoothing=label_smoothing,
    )


def train(
    model_name: str = "a",
    batch_size: int | None = None,
    epochs: int | None = None,
    steps_limit_per_epoch: int | None = None,
) -> dict:
    """
    Train one MS2 model with teacher forcing.
    Args:
        model_name: Model key, currently ``"a"`` or ``"b"``.
        batch_size: Batch size.
        epochs: Number of epochs.
        steps_limit_per_epoch: Optional cap for quick dry runs.
    Returns:
        Training summary dictionary.
    """
    ### validate supported models ###
    if model_name not in {"a", "b"}:
        raise ValueError("model_name must be 'a' or 'b'")

    ### resolve training settings from config with optional overrides ###
    training_config = get_model_training_config(model_name)
    resolved_batch_size = int(
        training_config.get("batch_size", 8) if batch_size is None else batch_size
    )
    resolved_epochs = int(
        training_config.get("epochs", 2) if epochs is None else epochs
    )
    pad_id = int(training_config.get("pad_id", 0))
    label_smoothing = float(training_config.get("label_smoothing", 0.1))

    ### load records and build datasets ###
    train_records = load_records("train")
    test_records = load_records("test")
    train_ds = build_dataset(
        records=train_records,
        batch_size=resolved_batch_size,
        shuffle=True,
    )
    test_ds = build_dataset(
        records=test_records,
        batch_size=resolved_batch_size,
        shuffle=False,
    )

    ### construct model and optimizer ###
    model: RNNModel | TransformerModel
    if model_name == "a":
        model = RNNModel(name="rnn_model_a")
    else:
        model = TransformerModel(name="transformer_model_b")
    steps_per_epoch = math.ceil(len(train_records) / max(resolved_batch_size, 1))
    if steps_limit_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, steps_limit_per_epoch)
    total_steps = max(steps_per_epoch * resolved_epochs, 1)
    optimizer = build_optimizer(model_name=model_name, total_steps=total_steps)

    ### optionally create checkpoint managers ###
    save_checkpoints = bool(training_config.get("save_checkpoints", True))
    best_manager: tf.train.CheckpointManager | None = None
    last_manager: tf.train.CheckpointManager | None = None
    best_eval_loss = float("inf")

    if save_checkpoints:
        best_dir, last_dir = build_checkpoint_dirs(model_name, training_config)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        best_manager = tf.train.CheckpointManager(
            checkpoint=checkpoint,
            directory=str(best_dir),
            max_to_keep=1,
            checkpoint_name="ckpt_best",
        )
        last_manager = tf.train.CheckpointManager(
            checkpoint=checkpoint,
            directory=str(last_dir),
            max_to_keep=1,
            checkpoint_name="ckpt_last",
        )

    ### run epochs ###
    history: list[dict] = []
    last_checkpoint_path = ""
    best_checkpoint_path = ""
    for epoch in range(resolved_epochs):
        ### train phase ###
        train_metric = tf.keras.metrics.Mean(name="train_loss")
        train_progbar = tf.keras.utils.Progbar(target=steps_per_epoch, verbose=1)
        for step, (features, targets) in enumerate(train_ds):
            if steps_limit_per_epoch is not None and step >= steps_limit_per_epoch:
                break
            loss = train_step(
                model_name,
                model,
                optimizer,
                features,
                targets,
                pad_id=pad_id,
                label_smoothing=label_smoothing,
            )
            train_metric.update_state(loss)
            train_progbar.update(step + 1, values=[("loss", float(loss.numpy()))])

        ### eval phase ###
        eval_metric = tf.keras.metrics.Mean(name="eval_loss")
        eval_steps = math.ceil(len(test_records) / max(resolved_batch_size, 1))
        eval_progbar = tf.keras.utils.Progbar(target=eval_steps, verbose=1)
        for eval_step_index, (features, targets) in enumerate(test_ds):
            loss = eval_step(
                model_name,
                model,
                features,
                targets,
                pad_id=pad_id,
                label_smoothing=label_smoothing,
            )
            eval_metric.update_state(loss)
            eval_progbar.update(eval_step_index + 1, values=[("eval_loss", float(loss.numpy()))])

        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": float(train_metric.result().numpy()),
            "eval_loss": float(eval_metric.result().numpy()),
        }
        history.append(epoch_summary)

        ### save last checkpoint every epoch ###
        if last_manager is not None:
            saved_last = last_manager.save(checkpoint_number=epoch + 1)
            last_checkpoint_path = "" if saved_last is None else saved_last

        ### save best checkpoint when eval improves ###
        current_eval = epoch_summary["eval_loss"]
        if best_manager is not None and current_eval < best_eval_loss:
            best_eval_loss = current_eval
            saved_best = best_manager.save(checkpoint_number=epoch + 1)
            best_checkpoint_path = "" if saved_best is None else saved_best

        print(
            f"[train] epoch={epoch_summary['epoch']} "
            f"train_loss={epoch_summary['train_loss']:.4f} "
            f"eval_loss={epoch_summary['eval_loss']:.4f}"
        )

        if save_checkpoints:
            print(
                f"[ckpt] epoch={epoch_summary['epoch']} "
                f"best={best_checkpoint_path or 'n/a'} "
                f"last={last_checkpoint_path or 'n/a'}"
            )

    return {
        "model": model,
        "history": history,
        "train_size": len(train_records),
        "test_size": len(test_records),
        "best_checkpoint": best_checkpoint_path,
        "last_checkpoint": last_checkpoint_path,
    }


if __name__ == "__main__":
    train(model_name="a", batch_size=4, epochs=1, steps_limit_per_epoch=3)
