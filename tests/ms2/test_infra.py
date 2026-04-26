from __future__ import annotations

import logging
import types
from pathlib import Path

import pytest

from src.common.paths import (
    list_ms2_cleaned_input_files,
    make_ms2_output_filename,
    resolve_ms2_paths,
)
from src.ms2.runtime import configure_runtime


def test_resolve_ms2_paths_creates_expected_dirs(tmp_path: Path) -> None:
    paths = resolve_ms2_paths(
        repo_root=tmp_path, model="model_b", seed=42, split="dev", create_dirs=True
    )

    assert paths.data_processed_ms2 == tmp_path / "data" / "processed" / "ms2"
    assert (
        paths.data_external_ms2_cleaned_input
        == tmp_path / "data" / "external" / "ms2-cleaned-input"
    )
    assert paths.experiments_ms2 == tmp_path / "experiments" / "ms2"
    assert paths.tokenizer_path.name == "ms2_tokenizer_bpe_4k_v001.model"
    assert paths.char_vocab_path.name == "ms2_char_vocab_v001.json"
    assert paths.tfrecord_shard_dir.name == "ms2_tfrecords_dev_v001.dir"
    assert paths.run_output_dir == tmp_path / "experiments" / "ms2" / "model_b" / "42"
    assert paths.report_dir == tmp_path / "docs" / "fs" / "ms2"

    for directory in (
        paths.data_processed_ms2,
        paths.experiments_ms2,
        paths.tfrecord_shard_dir,
        paths.run_output_dir,
        paths.report_dir,
    ):
        assert directory.is_dir()
    assert not paths.data_external_ms2_cleaned_input.exists()


def test_resolve_ms2_paths_can_skip_dir_creation(tmp_path: Path) -> None:
    paths = resolve_ms2_paths(repo_root=tmp_path, create_dirs=False)

    assert not paths.data_processed_ms2.exists()
    assert not paths.run_output_dir.exists()


def test_resolve_ms2_paths_manifest_matches_snapshot(tmp_path: Path) -> None:
    paths = resolve_ms2_paths(repo_root=tmp_path, create_dirs=False)

    assert paths.as_relative_manifest() == {
        "data_external_ms2_cleaned_input": "data/external/ms2-cleaned-input",
        "data_processed_ms2": "data/processed/ms2",
        "experiments_ms2": "experiments/ms2",
        "tokenizer_path": "data/processed/ms2/ms2_tokenizer_bpe_4k_v001.model",
        "char_vocab_path": "data/processed/ms2/ms2_char_vocab_v001.json",
        "tfrecord_shard_dir": "data/processed/ms2/ms2_tfrecords_train_v001.dir",
        "run_output_dir": "experiments/ms2/model_a/13",
        "report_dir": "docs/fs/ms2",
    }


def test_list_ms2_cleaned_input_files_reads_real_json_and_skips_examples(
    tmp_path: Path,
) -> None:
    paths = resolve_ms2_paths(repo_root=tmp_path, create_dirs=False)
    input_dir = paths.data_external_ms2_cleaned_input
    input_dir.mkdir(parents=True)
    real_input = input_dir / "lesson_001.json"
    another_real_input = input_dir / "lesson_002.json"
    real_input.write_text('{"data": []}\n', encoding="utf-8")
    another_real_input.write_text('{"data": []}\n', encoding="utf-8")
    (input_dir / "example.json.example").write_text('{"data": []}\n', encoding="utf-8")
    (input_dir / "example.txt.example").write_text("template\n", encoding="utf-8")
    (input_dir / "notes.txt").write_text("not an input\n", encoding="utf-8")

    assert list_ms2_cleaned_input_files(paths) == (real_input, another_real_input)


def test_list_ms2_cleaned_input_files_handles_missing_input_dir(tmp_path: Path) -> None:
    paths = resolve_ms2_paths(repo_root=tmp_path, create_dirs=False)

    assert list_ms2_cleaned_input_files(paths) == ()


def test_make_ms2_output_filename_contract() -> None:
    assert (
        make_ms2_output_filename("length", "analysis", 3, ".json")
        == "ms2_length_analysis_v003.json"
    )


@pytest.mark.parametrize(
    ("stage", "name", "version", "ext"),
    [
        ("bad-stage", "analysis", 1, "json"),
        ("length", "BadName", 1, "json"),
        ("length", "analysis", 0, "json"),
        ("length", "analysis", 1, ""),
        ("length", "analysis", 1, "json/../x"),
        ("length", "analysis", 1, "json\\x"),
    ],
)
def test_make_ms2_output_filename_rejects_invalid_tokens(
    stage: str, name: str, version: int, ext: str
) -> None:
    with pytest.raises(ValueError):
        make_ms2_output_filename(stage, name, version, ext)


def test_configure_runtime_sets_seed_policy_determinism_and_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[tuple[str, object]] = []

    fake_tf = types.SimpleNamespace(
        keras=types.SimpleNamespace(
            mixed_precision=types.SimpleNamespace(
                set_global_policy=lambda policy: calls.append(("policy", policy))
            ),
            utils=types.SimpleNamespace(
                set_random_seed=lambda seed: calls.append(("seed", seed))
            ),
        ),
        config=types.SimpleNamespace(
            experimental=types.SimpleNamespace(
                enable_op_determinism=lambda: calls.append(("determinism", True))
            )
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", fake_tf)

    with caplog.at_level(logging.INFO, logger="src.ms2.runtime"):
        summary = configure_runtime(seed=91)

    assert summary == {
        "seed": 91,
        "mixed_precision_policy": "mixed_float16",
        "deterministic_ops": True,
    }
    assert calls == [("policy", "mixed_float16"), ("seed", 91), ("determinism", True)]
    assert "Configured MS2 runtime" in caplog.text
    assert "seed=91" in caplog.text


def test_configure_runtime_can_disable_mixed_precision_and_determinism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    fake_tf = types.SimpleNamespace(
        keras=types.SimpleNamespace(
            mixed_precision=types.SimpleNamespace(
                set_global_policy=lambda policy: calls.append(("policy", policy))
            ),
            utils=types.SimpleNamespace(
                set_random_seed=lambda seed: calls.append(("seed", seed))
            ),
        ),
        config=types.SimpleNamespace(
            experimental=types.SimpleNamespace(
                enable_op_determinism=lambda: calls.append(("determinism", True))
            )
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", fake_tf)

    summary = configure_runtime(seed=13, mixed_precision=False, deterministic=False)

    assert summary == {
        "seed": 13,
        "mixed_precision_policy": "float32",
        "deterministic_ops": False,
    }
    assert calls == [("policy", "float32"), ("seed", 13)]


def test_ms2_modules_do_not_hardcode_canonical_relative_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ms2_files = sorted((repo_root / "src" / "ms2").rglob("*.py"))

    assert ms2_files
    for file_path in ms2_files:
        text = file_path.read_text(encoding="utf-8")
        assert "data/processed/ms2" not in text
        assert "experiments/ms2" not in text
