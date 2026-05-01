from __future__ import annotations

import importlib
import types
from pathlib import Path

import pytest

from src.ms2.schemas import RunConfig
from src.ms2.training.loop import train_one_run
from src.ms2.training.loss import label_smoothed_cross_entropy
from src.ms2.training.optimizer import build_adamw


def test_build_adamw_applies_loss_scale_and_decay_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAdamW:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeLossScaleOptimizer:
        def __init__(self, optimizer: object) -> None:
            self.inner = optimizer

    fake_tf = types.SimpleNamespace(
        keras=types.SimpleNamespace(
            optimizers=types.SimpleNamespace(AdamW=FakeAdamW),
            mixed_precision=types.SimpleNamespace(LossScaleOptimizer=FakeLossScaleOptimizer),
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", fake_tf)

    tracked = [types.SimpleNamespace(name="encoder/kernel:0"), types.SimpleNamespace(name="decoder/bias:0")]
    bundle = build_adamw(learning_rate=3e-4, tracked_variables=tracked)

    assert isinstance(bundle.optimizer, FakeLossScaleOptimizer)
    assert bundle.optimizer.inner.kwargs["weight_decay"] == 0.01
    assert bundle.optimizer.inner.kwargs["global_clipnorm"] == 1.0
    assert bundle.decay_variables == ("encoder/kernel:0",)
    assert bundle.teacher_forcing_ratio == 1.0
    assert bundle.gradient_clip_norm == 1.0


def test_schedules_subclass_learning_rate_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeScheduleBase:
        pass

    fake_tf = types.SimpleNamespace(
        keras=types.SimpleNamespace(
            optimizers=types.SimpleNamespace(
                schedules=types.SimpleNamespace(LearningRateSchedule=FakeScheduleBase)
            )
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", fake_tf)
    module = importlib.import_module("src.ms2.training.schedules")
    module = importlib.reload(module)

    cosine = module.CosineWithWarmup(total_steps=100)
    noam = module.Noam(d_model=256)

    assert isinstance(cosine, FakeScheduleBase)
    assert isinstance(noam, FakeScheduleBase)
    assert cosine(5) > 0
    assert noam(1) > 0


def test_schedule_values_follow_expected_shapes() -> None:
    from src.ms2.training.schedules import CosineWithWarmup, Noam

    cosine = CosineWithWarmup(total_steps=100, peak_lr=3e-4, min_lr=1e-5, warmup_ratio=0.05)
    assert cosine(1) < cosine(5)
    assert cosine(100) == pytest.approx(1e-5)

    noam = Noam(d_model=512, warmup_steps=1000)
    assert noam(10) > noam(1)
    assert noam(10000) < noam(1000)


def test_label_smoothed_cross_entropy_perfect_predictions_and_mask() -> None:
    logits = [[10.0, -10.0], [-10.0, 10.0]]
    targets = [0, 1]
    mask = [True, True]

    near_zero = label_smoothed_cross_entropy(logits, targets, mask, smoothing=0.0)
    assert near_zero < 1e-6

    masked_loss = label_smoothed_cross_entropy(logits, targets, [True, False], smoothing=0.0)
    changed_padded = label_smoothed_cross_entropy(logits, [0, 0], [True, False], smoothing=0.0)
    assert masked_loss == pytest.approx(changed_padded)


def test_train_one_run_writes_artifacts_and_respects_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.ms2.training.loop.Path.cwd", lambda: tmp_path)

    clock_values = iter([0.0, 0.1, 0.2, 0.3, 70.0, 70.1, 70.2, 70.3])
    monkeypatch.setattr("src.ms2.training.loop.time.monotonic", lambda: next(clock_values))

    class DummyModel:
        parameter_count = 123
        peak_gpu_memory_mb = 0.0
        mean_inference_time_ms_per_example = 1.5

        def train_step(self, _batch: object) -> float:
            return 0.25

        def evaluate(self, _dataset: object) -> dict[str, float]:
            return {"em": 0.5, "token_f1": 0.6, "char_edit_distance": 0.2, "bleu1": 0.4}

        def evaluate_test(self, _dataset: object) -> dict[str, float]:
            return {"em": 0.4, "token_f1": 0.55, "char_edit_distance": 0.25, "bleu1": 0.35}

    run_config = RunConfig(model_id="A", seed=13, wall_clock_budget_minutes=1)
    summary = train_one_run(DummyModel(), [1, 2, 3, 4], [1], run_config)

    run_dir = tmp_path / "experiments" / "ms2" / "model_a" / "13"
    assert summary.model_id == "A"
    assert summary.test_token_f1 == pytest.approx(0.55)
    assert (run_dir / "curves" / "train_step_losses.json").exists()
    assert (run_dir / "curves" / "dev_metrics.json").exists()
    assert (run_dir / "checkpoints" / "last.json").exists()
    assert (run_dir / "run_summary_v001.json").exists()
