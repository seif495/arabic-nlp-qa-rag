# AGENTS.md
Operational guide for coding agents working in this repository.

## 1) Scope and current repository state
- This repo is currently an architecture scaffold with docs and directory contracts.
- As of now, there are no committed Python source files, tests, or dependency manifests.
- Use this guide to keep future implementation consistent with the documented architecture.

## 2) Source-of-truth documents
- `README.md`
- `docs/decisions/adr_001_repo_structure.md` (ADR-001)
- `docs/fs/overall.md`
- `docs/fs/milestone-1.md`
- `docs/fs/milestone-2.md`
- `docs/fs/milestone-3.md`
- If this file conflicts with ADR-001, follow ADR-001 and update AGENTS.md.

## 3) Cursor and Copilot rules
- `.cursorrules`: not present.
- `.cursor/rules/`: not present.
- `.github/copilot-instructions.md`: not present.
- If any of these files are added later, merge their instructions into this document.

## 4) Environment and execution model
- Primary language/runtime: Python (implied by repo structure and CLI style).
- Entry points are CLI-first modules under `src/cli/`.
- Canonical command pattern:
  - `python -m src.cli.ms1 <command>`
  - `python -m src.cli.ms2 <command>`
  - `python -m src.cli.ms3 <command>`
- Keep CLI files thin and orchestration-focused.
- Keep reusable logic in `src/common/`, milestone logic in `src/ms1|ms2|ms3/`.

## 5) Build, lint, and test commands
Important: no official toolchain files are committed yet (`pyproject.toml`, `requirements*.txt`, `pytest.ini`, etc. are absent). Treat commands below as the expected convention once code is added.

### Setup (expected)
- `python -m venv .venv`
- `source .venv/bin/activate` (macOS/Linux)
- `pip install -r requirements.txt` (if present)
- `pip install -e .` (if package metadata is present)

### Build commands
- No dedicated build step exists yet.
- Packaging later: `python -m build`
- Bytecode sanity check: `python -m compileall src`

### Lint and format commands (expected)
- Format: `ruff format .`
- Lint: `ruff check .`
- Lint autofix: `ruff check . --fix`
- Type check: `mypy src`
- If Black/Flake8 is later adopted, follow committed config files.

### Test commands (pytest)
- Full suite: `pytest`
- Verbose: `pytest -vv`
- Stop on first failure: `pytest -x`
- File only: `pytest tests/test_<module>.py`
- Single test function: `pytest tests/test_<module>.py::test_<name>`
- Single test method: `pytest tests/test_<module>.py::Test<ClassName>::test_<name>`
- By keyword: `pytest -k "<expr>"`
- By marker: `pytest -m "<marker>"`

### Test commands (unittest fallback)
- Full suite: `python -m unittest discover -s tests -p "test_*.py"`
- Single test: `python -m unittest tests.test_module.TestClass.test_method`

## 6) Directory and ownership rules
- Treat this project as one cumulative system across milestones.
- Keep milestone-specific logic inside:
  - `src/ms1/`
  - `src/ms2/`
  - `src/ms3/`
- Put only truly reusable code in `src/common/`.
- Do not create vague folders like `misc`, `helpers`, or `tmp_utils`.
- Notebooks are exploratory only; production logic belongs in `src/`.

## 7) Code style guidelines

### Imports
- Group imports: standard library, third-party, local.
- Prefer absolute imports from package roots.
- Avoid wildcard imports.
- Import only what is used.
- Keep import side effects minimal.

### Formatting
- Follow PEP 8 with max line length 88 (or formatter default).
- Use one formatter repo-wide (prefer Ruff formatter if configured).
- Keep functions focused; extract helpers for complex branches.
- Avoid deeply nested control flow; prefer early returns/guards.

### Types and interfaces
- Add type hints to public functions and methods.
- Add explicit return type annotations.
- Use `dataclasses` or `TypedDict` for structured payloads.
- Use `Protocol`/ABC for pluggable components.
- Avoid `Any` unless unavoidable; narrow types at boundaries.

### Naming conventions
- `snake_case`: functions, variables, modules.
- `PascalCase`: classes.
- `UPPER_SNAKE_CASE`: constants.
- Use milestone-aware names when ambiguity is possible.
- Prefer descriptive names over short abbreviations.

### Error handling and logging
- Fail fast on invalid config and missing required files.
- Raise specific exceptions with actionable messages.
- Do not silently swallow exceptions.
- Validate external input at boundaries (CLI args, file IO, model responses).
- Use structured logging helpers from `src/common/logging_utils.py` once available.

### Configuration and secrets
- Store configuration in YAML under `configs/` only.
- Do not rely on `.env` for project configuration.
- Every non-sensitive config value needs an explicit default.
- Validate sensitive required values at startup and error clearly.

### Data handling
- Never modify `data/external/` in place.
- Write transformed outputs to `data/interim/`, `data/processed/`, or `data/artifacts/`.
- Keep filenames stable and meaningful (avoid `final2.csv` style names).
- Make milestone handoff artifacts explicit and reproducible.

### CLI conventions
- Expose key workflows through `src/cli/`.
- Keep CLI commands deterministic and scriptable.
- Return non-zero exit codes for failures.
- Print concise status, keep deep diagnostics in logs.

## 8) Testing conventions
- Mirror source structure under `tests/` where practical.
- Prefer unit tests for modules and integration tests for pipeline boundaries.
- Use fixtures for shared setup; avoid hidden global state.
- Seed randomness in tests for determinism.
- Add regression tests for bug fixes.

## 9) Documentation and change hygiene
- Update docs when behavior, commands, or contracts change.
- Keep milestone reports aligned with implementation details.
- In PRs/commits, explain why changes were made, not only what changed.
- Do not add generated artifacts to Git unless explicitly required.

## 10) Agent behavior checklist
- Before editing, read relevant FS/ADR docs.
- Respect module boundaries and existing naming patterns.
- Prefer minimal, targeted diffs.
- Run relevant lint/tests for touched code (or state clearly why not run).
- If toolchain files are added, update Section 5 with exact verified commands.
