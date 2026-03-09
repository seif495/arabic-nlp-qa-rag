from __future__ import annotations

import subprocess
import sys
import unittest
from unittest.mock import patch

from src.cli import ms1
from src.ms1.orchestration import CommandResult


class TestMS1CLICommands(unittest.TestCase):
    def test_required_single_commands_exist_and_run(self) -> None:
        commands = (
            "profile",
            "detect-irregularities",
            "normalize",
            "build-dataset",
        )
        for command in commands:
            with self.subTest(command=command):
                completed = subprocess.run(
                    [sys.executable, "-m", "src.cli.ms1", command],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0)
                self.assertIn(f"[{command}] status=ok", completed.stdout)

    def test_run_all_exists_and_runs(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "src.cli.ms1", "run-all"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("[profile] status=ok", completed.stdout)
        self.assertIn("[detect-irregularities] status=ok", completed.stdout)
        self.assertIn("[normalize] status=ok", completed.stdout)
        self.assertIn("[build-dataset] status=ok", completed.stdout)


class TestMS1CLIDelegation(unittest.TestCase):
    def test_main_delegates_to_orchestration_handler(self) -> None:
        with (
            patch.object(sys, "argv", ["ms1", "profile"]),
            patch(
                "src.cli.ms1.orchestration.profile",
                return_value=CommandResult(
                    command="profile",
                    status="ok",
                    output_path="/tmp/profile",
                ),
            ) as mock_profile,
        ):
            exit_code = ms1.main()

        self.assertEqual(exit_code, 0)
        mock_profile.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
