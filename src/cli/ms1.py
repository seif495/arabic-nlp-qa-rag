from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from src.ms1 import orchestration
from src.ms1.orchestration import CommandResult

SingleCommandHandler = Callable[[Path | None], CommandResult]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.ms1",
        description="Milestone 1 pipeline command line interface.",
    )
    parser.add_argument(
        "command",
        choices=(
            "profile",
            "detect-irregularities",
            "normalize",
            "build-dataset",
            "run-all",
        ),
        help="Milestone 1 command to execute.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    handlers: dict[str, SingleCommandHandler] = {
        "profile": orchestration.profile,
        "detect-irregularities": orchestration.detect_irregularities,
        "normalize": orchestration.normalize,
        "build-dataset": orchestration.build_dataset,
    }

    if args.command == "run-all":
        for result in orchestration.run_all(repo_root=repo_root):
            _print_result(result)
        return 0

    result = handlers[args.command](repo_root)
    _print_result(result)
    return 0


def _print_result(result: CommandResult) -> None:
    print(f"[{result.command}] status={result.status} output={result.output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
