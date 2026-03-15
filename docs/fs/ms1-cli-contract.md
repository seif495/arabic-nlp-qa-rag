# MS1 CLI Contract

This document defines the Milestone 1 command-line interface exposed through
`python -m src.cli.ms1`.

## Supported Commands

```bash
python -m src.cli.ms1 profile
python -m src.cli.ms1 detect-irregularities
python -m src.cli.ms1 normalize
python -m src.cli.ms1 build-dataset
python -m src.cli.ms1 run-all
```

## Behavioral Contract

- The CLI parses a single required positional command.
- The CLI delegates execution to `src.ms1.orchestration` handlers.
- Command handlers return structured results (`command`, `status`, `output_path`).
- `run-all` executes all command handlers in this order:
  1. `profile`
  2. `detect-irregularities`
  3. `normalize`
  4. `build-dataset`
- CLI output prints one status line per executed command.

## Example Output

```text
[profile] status=ok output=<repo>/experiments/ms1/profile
[detect-irregularities] status=ok output=<repo>/experiments/ms1/irregularities
[normalize] status=ok output=<repo>/data/interim/normalized
[build-dataset] status=ok output=<repo>/data/processed/ms1
```
