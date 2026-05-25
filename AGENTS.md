# AGENTS.md

Compact, repo-specific guidance for coding agents.

## Current Reality (do not assume scaffold-only)
- This repo already has working Python code and tests for Milestone 1 under `src/ms1/`, `src/common/`, `src/cli/ms1.py`, and `tests/`.
- `src/ms2/` and `src/ms3/` are directory scaffolds only (no Python package modules/entrypoints yet).
- If docs claim something is "planned" but code/config exists, trust code/config first.

## Verified Toolchain and Commands
- Python version is pinned by `.python-version` to `3.11`.
- Use `uv` workflow (lockfile is committed): `uv sync --dev`.
- Lint/format: `uv run ruff check .` and `uv run ruff format .`.
- Tests: `uv run pytest`.
- Run one test file: `uv run pytest tests/ms1/test_dataset_export.py`.
- Run one test: `uv run pytest tests/ms1/test_dataset_export.py::TestDatasetExport::test_sanity_check_reports_usable_dataset`.

## Entrypoints and Packaging Gotchas
- Only implemented CLI is MS1: `uv run python -m src.cli.ms1 <command>` where `<command>` is one of `profile`, `detect-irregularities`, `normalize`, `build-dataset`, `run-all`.
- Installed script `ms1` exists via `pyproject.toml` (`[project.scripts] ms1 = "src.cli.ms1:main"`).
- `setuptools` package list is explicit (`src`, `src.cli`, `src.ms1`, `src.common`); if you add importable packages (for example `src.ms2`/`src.ms3` with `__init__.py`), update `pyproject.toml` or packaging/tests will drift.

## Data + Path Contracts Enforced in Code
- Do not edit `data/external/` in place.
- MS1 outputs should go through `src/common/paths.py` (`resolve_ms1_paths`, `MS1Paths`) instead of hardcoded paths.
- Filename contract is enforced by `make_ms1_output_filename`: `ms1_<stage>_<name>_v###.<ext>`, lowercase snake_case tokens only.
- `resolve_ms1_paths(..., create_dirs=True)` auto-creates writable output dirs (`data/interim`, `data/processed/ms1`, `experiments/ms1`, `docs/reports`).

## Tests That Catch Easy-to-Miss Regressions
- `tests/common/test_schemas.py` keeps `src/common/schemas.py` EXAMPLE constants in sync with files under `docs/fs/artifacts/ms1/`; update both together.
- `tests/cli/test_ms1_cli.py` checks CLI output pattern (`[<command>] status=...`) and command order for `run-all`.
- `build_processed_dataset_records` only loads QA files matching `data/external/qa/*_QA.csv`; transcript context matching is title-based after canonicalization in `src/ms1/dataset_export.py`.

## High-Value References
- `pyproject.toml` (actual commands, scripts, package boundaries)
- `src/cli/ms1.py` and `src/ms1/orchestration.py` (real execution flow)
- `src/common/paths.py` (path/output contract)
- `docs/fs/artifacts/ms1/ms1-artifact-map.md` (artifact naming/path intent)
