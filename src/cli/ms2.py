from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.ms2 import orchestration
from src.ms2.orchestration import CommandResult


MODEL_CHOICES = ("a", "b")
SEED_CHOICES = (13, 42, 91)
SPLIT_CHOICES = ("dev", "test")
DECODING_CHOICES = ("greedy", "beam")
ABLATION_VARIANTS = (
    "no_film",
    "mean_merge",
    "plain_branch3",
    "sinusoidal_pe",
    "no_pe",
    "shared_layers",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.ms2",
        description="Milestone 2 pipeline command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep_data = subparsers.add_parser(
        "prep-data",
        help="Train tokenizer, build char vocab, and cache TFRecords.",
    )
    prep_data.add_argument(
        "--target-model",
        choices=MODEL_CHOICES,
        default="a",
        help="Pipeline schema target for cached examples.",
    )

    subparsers.add_parser(
        "analyze-lengths",
        help="Analyze MS1 output lengths and freeze MS2 caps.",
    )

    train = subparsers.add_parser(
        "train",
        help="Train one model on one seed from a RunConfig JSON file.",
    )
    _add_model_seed_args(train)
    train.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to RunConfig JSON for parameter-heavy settings.",
    )

    infer = subparsers.add_parser(
        "infer",
        help="Run greedy or beam decoding for a trained run.",
    )
    _add_model_seed_args(infer)
    infer.add_argument("--split", choices=SPLIT_CHOICES, default="dev")
    infer.add_argument("--decoding", choices=DECODING_CHOICES, default="greedy")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Compute EM, F1, char edit distance, and BLEU-1.",
    )
    _add_model_seed_args(evaluate)
    evaluate.add_argument("--split", choices=SPLIT_CHOICES, default="dev")

    subparsers.add_parser(
        "evaluate-protocol",
        help="Run the full MS2 evaluation protocol battery.",
    )

    ablate = subparsers.add_parser(
        "ablate",
        help="Run a configured ablation variant.",
    )
    ablate.add_argument("--variant", choices=ABLATION_VARIANTS, default="no_film")
    ablate.add_argument("--seed", type=int, choices=SEED_CHOICES, default=13)

    subparsers.add_parser(
        "compare",
        help="Build the MS2 comparison table and diagnostic plots.",
    )
    run_all = subparsers.add_parser(
        "run-all",
        help="Run the full MS2 milestone orchestration in dependency order.",
    )
    run_all.add_argument(
        "--force-from",
        "--force",
        dest="force_from",
        choices=orchestration.RUN_ALL_STAGE_ORDER,
        default=None,
        help="Force re-execution starting from the selected stage.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path(
        os.environ.get("MS2_REPO_ROOT", str(Path(__file__).resolve().parents[2]))
    ).resolve()

    try:
        if args.command == "run-all":
            for result in orchestration.run_all(
                repo_root=repo_root,
                force_from=args.force_from,
            ):
                _print_result(result)
            return 0

        result = _dispatch(args, repo_root)
        _print_result(result)
        return 0
    except Exception as exc:
        print(f"[{args.command}] status=error error={exc}", file=sys.stderr)
        return 1


def _add_model_seed_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=MODEL_CHOICES, default="a")
    parser.add_argument("--seed", type=int, choices=SEED_CHOICES, default=13)


def _dispatch(args: argparse.Namespace, repo_root: Path) -> CommandResult:
    if args.command == "analyze-lengths":
        return orchestration.analyze_lengths(repo_root=repo_root)
    if args.command == "prep-data":
        return orchestration.prep_data(
            repo_root=repo_root,
            target_model=args.target_model,
        )
    if args.command == "train":
        return orchestration.train(
            repo_root=repo_root,
            model=args.model,
            seed=args.seed,
            config=args.config,
        )
    if args.command == "infer":
        return orchestration.infer(
            repo_root=repo_root,
            model=args.model,
            seed=args.seed,
            split=args.split,
            decoding=args.decoding,
        )
    if args.command == "evaluate":
        return orchestration.evaluate(
            repo_root=repo_root,
            model=args.model,
            seed=args.seed,
            split=args.split,
        )
    if args.command == "evaluate-protocol":
        return orchestration.evaluate_protocol(repo_root=repo_root)
    if args.command == "ablate":
        return orchestration.ablate(
            repo_root=repo_root,
            variant=args.variant,
            seed=args.seed,
        )
    if args.command == "compare":
        return orchestration.compare(repo_root=repo_root)
    raise ValueError(f"Unsupported command: {args.command}")


def _print_result(result: CommandResult) -> None:
    message = (
        f"[{result.command}] status={result.status} "
        f"output={result.output_path}"
    )
    if result.elapsed_seconds is not None:
        message = f"{message} wall_clock_seconds={result.elapsed_seconds:.6f}"
    if result.budget_warning:
        message = f"{message} warning={result.budget_warning}"
    print(message)


if __name__ == "__main__":
    raise SystemExit(main())
