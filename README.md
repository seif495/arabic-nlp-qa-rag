# Arabic NLP QA RAG

End-to-end Arabic NLP project for Spring 2026: from noisy raw transcripts to a retrieval-augmented QA system with semantic caching.

This repository is intentionally organized as one cumulative system across three milestones:
- Milestone 1: dataset understanding and Arabic text preparation.
- Milestone 2: RNN vs Transformer modeling from scratch.
- Milestone 3: RAG + semantic cache + prompt/context strategy experiments.

## Why This Repo Exists

The project objective is not only to produce outputs, but to demonstrate design reasoning, implementation clarity, and trade-off analysis in a discussion-based evaluation.

In practice, that means this repo must always support:
- fast onboarding for teammates,
- reproducible CLI-first execution,
- clear milestone boundaries with cumulative progression,
- auditable decisions and artifacts.

## Quick Start

1. Clone the repository.
2. Add course dataset files under `data/external/` using the expected structure.
3. Implement and run milestone pipelines through `src/cli/` entrypoints.
4. Track experiments in `experiments/` and document results in `docs/reports/`.

Current status: the repository already contains the full architecture scaffold and directory contracts; implementation modules are added milestone by milestone.

## Project Scope

Dataset source and shape:
- 13 Arabic transcript files (`.txt`) from ElDa7ee7 Season 8.
- 13 QA CSV files (300 extractive pairs each).
- Total QA pairs: 3900.
- CSV schema: `video_id, video_title, question_id, question, answer, difficulty`.

Language characteristics the system must handle:
- MSA + Egyptian dialect mix.
- conversational narration and rhetorical structure.
- Arabic-English code-switching.
- orthographic inconsistency and noisy punctuation.
- named-entity spelling variation and long context shifts.

## Architecture At A Glance

This repo follows ADR-001 (`docs/decisions/adr_001_repo_structure.md`):
- one repository, one evolving system,
- shared logic in `src/common/`,
- milestone-specific logic isolated in `src/ms1/`, `src/ms2/`, `src/ms3/`,
- orchestration through `src/cli/`.

Top-level structure:
- `src/` core implementation by milestone and shared modules.
- `configs/` YAML configuration sharded by milestone.
- `data/` immutable raw inputs + staged outputs/artifacts.
- `experiments/` trial runs and ablations.
- `notebooks/` exploration only (not production logic).
- `docs/fs/` functional specs and project framing.
- `docs/decisions/` architecture decisions (ADRs).
- `docs/reports/` milestone report submissions.
- `docs/tickets/` milestone ticket board and developer assignments.

## Milestone Contracts

### Milestone 1: Understanding and Preparing Arabic Text

Goal:
- profile the dataset,
- identify hidden patterns and irregularities,
- normalize Arabic text and optional spelling corrections,
- export prepared data for downstream neural modeling.

Primary module areas:
- `src/ms1/ingest/`
- `src/ms1/profiling/`
- `src/ms1/cleaning/`
- `src/ms1/normalization/`
- `src/ms1/spelling/`
- `src/ms1/tokenization/`
- `src/ms1/dataset_build/`
- `src/ms1/evaluation/`

### Milestone 2: RNN vs Transformer (From Scratch)

Goal:
- build Model A (RNN-variant) and Model B (simple Transformer), both without pretrained weights,
- compare behavior and trade-offs, not just raw scores.

Required comparisons:
- performance,
- convergence speed,
- parameter count,
- training stability,
- sensitivity to noise,
- generalization.

Primary module areas:
- `src/ms2/models/rnn/`
- `src/ms2/models/transformer/`
- `src/ms2/training/`
- `src/ms2/metrics/`
- `src/ms2/analysis/`
- `src/ms2/inference/`

### Milestone 3: RAG + Semantic Cache + Prompt/Context Experiments

Goal:
- implement vanilla RAG over cleaned Arabic data,
- add semantic caching based on embedding similarity,
- evaluate prompt designs and context window strategies.

Required analyses include:
- cache hit rate, saved model calls, threshold sensitivity,
- prompt-guided vs no-system behaviors,
- context strategy trade-offs in latency, coherence, token use, and accuracy.

Primary module areas:
- `src/ms3/retrieval/`
- `src/ms3/generation/`
- `src/ms3/cache/`
- `src/ms3/prompting/`
- `src/ms3/context_strategies/`
- `src/ms3/evaluation/`

## Execution Model (CLI-First)

All major workflows must be exposed through `src/cli/`. Core logic belongs in milestone modules; CLI files should orchestrate only.

Canonical command style:

```bash
python -m src.cli.ms1 <command>
python -m src.cli.ms2 <command>
python -m src.cli.ms3 <command>
```

Examples (target interface):

```bash
python -m src.cli.ms1 profile
python -m src.cli.ms1 build-dataset
python -m src.cli.ms2 train-rnn
python -m src.cli.ms2 evaluate-transformer
python -m src.cli.ms3 build-index
python -m src.cli.ms3 run-rag
```

## Data and Artifact Rules

These are mandatory project rules:
- Never modify raw files in `data/external/` in place.
- Write transformed outputs to `data/interim/`, `data/processed/`, or `data/artifacts/`.
- Keep artifact handoffs explicit between milestones.
- Use stable, non-ambiguous names for generated assets.
- Centralize path handling and config loading in shared utilities.

## Configuration Rules

- Configuration is YAML-only under `configs/`.
- No `.env`-driven project configuration.
- Non-sensitive fields must always have explicit defaults.
- Sensitive required values must fail fast with clear startup errors.

## Branching and Collaboration

Long-lived branches:
- `main`
- `milestone-1`
- `milestone-2`
- `milestone-3`

Short-lived feature branch examples:
- `feat/ms1-profiling`
- `feat/ms2-rnn`
- `feat/ms3-indexing`

Collaboration principle:
- parallel work by module ownership,
- minimal collision in shared orchestration files,
- reports and code terminology must stay consistent.

## Evaluation Deliverables

For each milestone:
- code in the corresponding milestone branch,
- 2-page technical report (`.md`) on that branch,
- one-to-one discussion evaluation.

Grading context:
- each milestone: 8% of course grade,
- in-class tasks and participation: 6%.

Deadlines:
- Milestone 1: 14 March 2026, 11:59 PM.
- Milestone 2: 22 April 2026, 11:59 PM.
- Milestone 3: 16 May 2026, 11:59 PM.

Evaluation snapshot is the last commit before each deadline.

## Core References

- Project description: `docs/fs/project-desc.md`
- Overall functional spec: `docs/fs/overall.md`
- Ticket board index: `docs/tickets/README.md`
- Milestone specs:
  - `docs/fs/milestone-1.md`
  - `docs/fs/milestone-2.md`
  - `docs/fs/milestone-3.md`
- Milestone ticket files:
  - `docs/tickets/ms1/roadmap.md`
  - `docs/tickets/ms2/roadmap.md`
  - `docs/tickets/ms3/roadmap.md`
- Repository ADR: `docs/decisions/adr_001_repo_structure.md`
