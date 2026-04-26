from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

from src.cli import ms2
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

RUN_ALL_ORDER = (
    "analyze-lengths",
    "prep-data",
    "train",
    "infer",
    "evaluate",
    "evaluate-protocol",
    "ablate",
    "compare",
)


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
        for command in COMMANDS:
            with self.subTest(command=command):
                completed = subprocess.run(
                    [sys.executable, "-m", "src.cli.ms2", command],
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(completed.returncode, 0)
                self.assertIn(f"[{command}] status=ok", completed.stdout)

    def test_run_all_exists_and_runs_in_dependency_order(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "src.cli.ms2", "run-all"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0)
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        self.assertEqual(len(lines), len(RUN_ALL_ORDER))
        for line, command in zip(lines, RUN_ALL_ORDER, strict=True):
            self.assertIn(f"[{command}] status=ok", line)

    def test_console_script_registered(self) -> None:
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('ms2 = "src.cli.ms2:main"', pyproject)


class TestMS2CLIDelegation(unittest.TestCase):
    def test_main_delegates_to_orchestration_handler(self) -> None:
        with (
            patch.object(sys, "argv", ["ms2", "train"]),
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
            config=None,
        )
        mock_print.assert_called_once()

    def test_main_delegates_run_all_to_orchestration_order(self) -> None:
        results = [
            CommandResult(command=command, status="ok", output_path=f"/tmp/{command}")
            for command in RUN_ALL_ORDER
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
        mock_run_all.assert_called_once_with(repo_root=ANY)
        printed = [call.args[0] for call in mock_print.call_args_list]
        self.assertEqual(
            printed,
            [f"[{command}] status=ok output=/tmp/{command}" for command in RUN_ALL_ORDER],
        )

    def test_main_returns_non_zero_on_handler_error(self) -> None:
        with (
            patch.object(sys, "argv", ["ms2", "train"]),
            patch("src.cli.ms2.orchestration.train", side_effect=RuntimeError("boom")),
        ):
            exit_code = ms2.main()

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
