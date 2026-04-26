from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import ANY, patch

from src.cli import ms2
from src.ms2 import orchestration
from src.ms2.orchestration import CommandResult


COMMANDS = (
    "prep-data",
    "analyze-lengths",
    "train",
    "infer",
    "evaluate",
    "evaluate-protocol",
    "ablate",
    "compare",
)

RUN_ALL_COUNTS = {
    "analyze-lengths": 1,
    "prep-data": 2,
    "train": 6,
    "infer": 6,
    "evaluate": 6,
    "evaluate-protocol": 1,
    "ablate": 3,
    "compare": 1,
}


class TestMS2CLICommands(unittest.TestCase):
    def test_required_commands_emit_help(self) -> None:
        for command in (*COMMANDS, "run-all"):
            with self.subTest(command=command):
                completed = subprocess.run(
                    [sys.executable, "-m", "src.cli.ms2", command, "--help"],
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(completed.returncode, 0)
                self.assertIn("usage: python -m src.cli.ms2", completed.stdout)

    def test_required_single_commands_exist_and_run(self) -> None:
        command_args = {
            "train": [
                "--config",
                "docs/fs/artifacts/ms2/ms2-run-config.example.json",
            ]
        }
        for command in COMMANDS:
            with self.subTest(command=command):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "src.cli.ms2",
                        command,
                        *command_args.get(command, []),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(completed.returncode, 0)
                self.assertIn(f"[{command}] status=ok", completed.stdout)

    def test_train_requires_config(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "src.cli.ms2", "train"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--config", completed.stderr)

    def test_run_all_exists_and_runs_in_dependency_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_repo_root:
            env = os.environ.copy()
            env["MS2_REPO_ROOT"] = temp_repo_root
            completed = subprocess.run(
                [sys.executable, "-m", "src.cli.ms2", "run-all"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertEqual(completed.returncode, 0)
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        commands = [line.split("]", maxsplit=1)[0].lstrip("[") for line in lines]
        self.assertEqual(Counter(commands), RUN_ALL_COUNTS)
        self.assertTrue(all("wall_clock_seconds=" in line for line in lines))
        self.assertTrue(lines[0].startswith("[analyze-lengths] status="))

    def test_console_script_registered(self) -> None:
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('ms2 = "src.cli.ms2:main"', pyproject)


class TestMS2CLIDelegation(unittest.TestCase):
    def test_main_delegates_to_orchestration_handler(self) -> None:
        with (
            patch.object(sys, "argv", ["ms2", "train", "--config", "config.json"]),
            patch(
                "src.cli.ms2.orchestration.train",
                return_value=CommandResult(
                    command="train",
                    status="ok",
                    output_path="/tmp/train",
                ),
            ) as mock_train,
            patch("builtins.print") as mock_print,
        ):
            exit_code = ms2.main()

        self.assertEqual(exit_code, 0)
        mock_train.assert_called_once_with(
            repo_root=ANY,
            model="a",
            seed=13,
            config=Path("config.json"),
        )
        mock_print.assert_called_once()

    def test_main_forwards_prep_data_args(self) -> None:
        with (
            patch.object(sys, "argv", ["ms2", "prep-data", "--target-model", "b"]),
            patch(
                "src.cli.ms2.orchestration.prep_data",
                return_value=CommandResult("prep-data", "ok", "/tmp/prep"),
            ) as mock_prep_data,
            patch("builtins.print"),
        ):
            exit_code = ms2.main()

        self.assertEqual(exit_code, 0)
        mock_prep_data.assert_called_once_with(repo_root=ANY, target_model="b")

    def test_main_forwards_infer_args(self) -> None:
        with (
            patch.object(
                sys,
                "argv",
                [
                    "ms2",
                    "infer",
                    "--model",
                    "b",
                    "--seed",
                    "91",
                    "--split",
                    "test",
                    "--decoding",
                    "beam",
                ],
            ),
            patch(
                "src.cli.ms2.orchestration.infer",
                return_value=CommandResult("infer", "ok", "/tmp/infer"),
            ) as mock_infer,
            patch("builtins.print"),
        ):
            exit_code = ms2.main()

        self.assertEqual(exit_code, 0)
        mock_infer.assert_called_once_with(
            repo_root=ANY,
            model="b",
            seed=91,
            split="test",
            decoding="beam",
        )

    def test_main_forwards_evaluate_args(self) -> None:
        with (
            patch.object(
                sys,
                "argv",
                ["ms2", "evaluate", "--model", "b", "--seed", "42", "--split", "test"],
            ),
            patch(
                "src.cli.ms2.orchestration.evaluate",
                return_value=CommandResult("evaluate", "ok", "/tmp/evaluate"),
            ) as mock_evaluate,
            patch("builtins.print"),
        ):
            exit_code = ms2.main()

        self.assertEqual(exit_code, 0)
        mock_evaluate.assert_called_once_with(
            repo_root=ANY,
            model="b",
            seed=42,
            split="test",
        )

    def test_main_forwards_ablate_args(self) -> None:
        with (
            patch.object(
                sys,
                "argv",
                ["ms2", "ablate", "--variant", "shared_layers", "--seed", "91"],
            ),
            patch(
                "src.cli.ms2.orchestration.ablate",
                return_value=CommandResult("ablate", "ok", "/tmp/ablate"),
            ) as mock_ablate,
            patch("builtins.print"),
        ):
            exit_code = ms2.main()

        self.assertEqual(exit_code, 0)
        mock_ablate.assert_called_once_with(
            repo_root=ANY,
            variant="shared_layers",
            seed=91,
        )

    def test_main_delegates_no_arg_commands(self) -> None:
        command_handlers = (
            ("analyze-lengths", "analyze_lengths"),
            ("evaluate-protocol", "evaluate_protocol"),
            ("compare", "compare"),
        )
        for command, handler_name in command_handlers:
            with self.subTest(command=command):
                with (
                    patch.object(sys, "argv", ["ms2", command]),
                    patch(
                        f"src.cli.ms2.orchestration.{handler_name}",
                        return_value=CommandResult(command, "ok", f"/tmp/{command}"),
                    ) as mock_handler,
                    patch("builtins.print"),
                ):
                    exit_code = ms2.main()

                self.assertEqual(exit_code, 0)
                mock_handler.assert_called_once_with(repo_root=ANY)

    def test_main_delegates_run_all_to_orchestration_order(self) -> None:
        results = [
            CommandResult(
                command=command,
                status="ok",
                output_path=f"/tmp/{command}",
                elapsed_seconds=0.5,
            )
            for command in orchestration.RUN_ALL_STAGE_ORDER
        ]
        with (
            patch.object(sys, "argv", ["ms2", "run-all"]),
            patch(
                "src.cli.ms2.orchestration.run_all",
                return_value=results,
            ) as mock_run_all,
            patch("builtins.print") as mock_print,
        ):
            exit_code = ms2.main()

        self.assertEqual(exit_code, 0)
        mock_run_all.assert_called_once_with(repo_root=ANY, force_from=None)
        printed = [call.args[0] for call in mock_print.call_args_list]
        self.assertEqual(
            printed,
            [
                f"[{command}] status=ok output=/tmp/{command} wall_clock_seconds=0.500000"
                for command in orchestration.RUN_ALL_STAGE_ORDER
            ],
        )

    def test_main_forwards_run_all_force_from(self) -> None:
        with (
            patch.object(sys, "argv", ["ms2", "run-all", "--force-from", "train"]),
            patch(
                "src.cli.ms2.orchestration.run_all",
                return_value=[
                    CommandResult(
                        command="train",
                        status="ok",
                        output_path="/tmp/train",
                        elapsed_seconds=0.2,
                    )
                ],
            ) as mock_run_all,
            patch("builtins.print"),
        ):
            exit_code = ms2.main()

        self.assertEqual(exit_code, 0)
        mock_run_all.assert_called_once_with(repo_root=ANY, force_from="train")

    def test_main_returns_non_zero_on_handler_error(self) -> None:
        with (
            patch.object(sys, "argv", ["ms2", "train", "--config", "config.json"]),
            patch("src.cli.ms2.orchestration.train", side_effect=RuntimeError("boom")),
        ):
            exit_code = ms2.main()

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
