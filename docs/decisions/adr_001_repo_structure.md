# ADR-001: Repository Structure and Development Conventions

## Status

Accepted

## Date

2026-03-06

## Decision

The project will use a **single repository** organized around a **shared core** and **milestone-specific vertical slices**.

The structure must satisfy two constraints simultaneously:

1. **Milestones are sequential and cumulative**  
   Milestone 2 builds on Milestone 1, and Milestone 3 builds on the previous milestones.

2. **Development within a milestone must support parallel work**  
   Multiple developers must be able to work concurrently with minimal file collision and clear ownership boundaries.

Accordingly, the canonical repository structure is:

```text
repo/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── fs/
│   │   ├── overall.md
│   │   ├── milestone-1.md
│   │   ├── milestone-2.md
│   │   └── milestone-3.md
│   ├── reports/
│   │   ├── ms1_report.md
│   │   ├── ms2_report.md
│   │   └── ms3_report.md
│   ├── tickets/
│   │   ├── README.md
│   │   ├── ms1/
│   │   │   ├── roadmap.md
│   │   │   ├── alice.md
│   │   │   ├── bob.md
│   │   │   └── charly.md
│   │   ├── ms2/
│   │   │   ├── roadmap.md
│   │   │   ├── alice.md
│   │   │   ├── bob.md
│   │   │   └── charly.md
│   │   └── ms3/
│   │       ├── roadmap.md
│   │       ├── alice.md
│   │       ├── bob.md
│   │       └── charly.md
│   └── decisions/
│       └── adr_001_repo_structure.md
├── data/
│   ├── external/
│   │   ├── transcripts/
│   │   └── qa/
│   ├── interim/
│   │   ├── normalized/
│   │   ├── tokenized/
│   │   └── splits/
│   ├── processed/
│   │   ├── ms1/
│   │   ├── ms2/
│   │   └── ms3/
│   └── artifacts/
│       ├── vocab/
│       ├── embeddings/
│       ├── indexes/
│       └── cache/
├── experiments/
│   ├── ms1/
│   ├── ms2/
│   └── ms3/
├── notebooks/
│   ├── ms1/
│   ├── ms2/
│   └── ms3/
├── configs/
│   ├── general/
│   ├── ms1/
│   ├── ms2/
│   └── ms3/
└── src/
    ├── common/
    │   ├── io.py
    │   ├── paths.py
    │   ├── constants.py
    │   ├── logging_utils.py
    │   ├── seed.py
    │   ├── schemas.py
    │   └── config_loader.py
    ├── ms1/
    │   ├── ingest/
    │   ├── profiling/
    │   ├── cleaning/
    │   ├── normalization/
    │   ├── spelling/
    │   ├── tokenization/
    │   ├── dataset_build/
    │   └── evaluation/
    ├── ms2/
    │   ├── data/
    │   ├── models/
    │   │   ├── rnn/
    │   │   └── transformer/
    │   ├── training/
    │   ├── inference/
    │   ├── metrics/
    │   └── analysis/
    ├── ms3/
    │   ├── retrieval/
    │   ├── indexing/
    │   ├── generation/
    │   ├── cache/
    │   ├── prompting/
    │   ├── context_strategies/
    │   └── evaluation/
    └── cli/
        ├── ms1.py
        ├── ms2.py
        └── ms3.py
```

## Architectural Rules

### 1. One repository, one evolving system

The project is treated as one cumulative system, not three disconnected mini-projects.

### 2. Shared logic goes into src/common

Only genuinely reusable code belongs in src/common.
Examples include path handling, IO helpers, shared schemas, logging, and reproducibility utilities.

### 3. Milestone-specific logic stays inside its milestone

If a module exists primarily to satisfy one milestone, it belongs under that milestone’s folder.

### 4. No generic dumping folders

Avoid vague folders such as misc, helpers, or milestone-level utils unless the purpose is highly specific and justified.

### 5. Raw data is immutable

Files inside data/external/ must not be edited in place.
All transformations must produce new outputs under interim, processed, or artifacts.

### 6. Artifact handoff must be explicit

Outputs of one stage must be saved explicitly and reused by later stages instead of being recreated implicitly through hidden scripts.

### 7. Notebooks are exploratory only

Notebooks may be used for exploration, debugging, and visualization, but core logic must live in src/.

### 8. Configurations are YAML-only and centrally loaded

The project will not use `.env` files.
All configuration values must be stored as YAML files under `configs/` and loaded through a dedicated shared loader script in `src/common/config_loader.py`.

The configuration space is sharded into four scopes:

- `configs/general/`
- `configs/ms1/`
- `configs/ms2/`
- `configs/ms3/`

Default values are non-negotiable: every non-sensitive configuration field must define an explicit default value.
Any required sensitive value (for example, API keys, tokens, or credentials) must be validated at startup and fail fast with a clear, actionable error message when missing or invalid.

### 9. Experiments are not production code

Experimental outputs, trial runs, and ablation artifacts belong in experiments/, not in src/ or mixed into data folders.

### 9.1 Ticket tracking is documentation-first

Task planning and developer ownership are tracked in markdown under `docs/tickets/`.

Each milestone folder must contain exactly four files:

- `roadmap.md`
- `alice.md`
- `bob.md`
- `charly.md`

This keeps planning auditable in-repo without relying on external project management tools.

### 10. Execution must be standardized through CLI entrypoints

All major pipeline actions must be callable through the project CLI.
Ad hoc execution through random scripts is not acceptable as the primary interface.

## Mandatory Best Practices

### CLI-first execution

All important workflows must be exposed through src/cli/.

Examples:

- preprocessing
- dataset building
- training
- evaluation
- index building
- cache generation
- retrieval pipeline execution

This ensures a consistent interface for all developers and reduces hidden execution logic.

#### Example command style:

```bash
python -m src.cli.ms1 profile
python -m src.cli.ms1 build-dataset
python -m src.cli.ms2 train-rnn
python -m src.cli.ms2 evaluate-transformer
python -m src.cli.ms3 build-index
python -m src.cli.ms3 run-rag
```

Thin CLI, thick modules

CLI files should only orchestrate.
Core logic must remain inside milestone modules.

### Reproducibility by default

Runs should be reproducible through:

- fixed seeds
- explicit configs
- stable path resolution
- deterministic saved outputs where possible

### Clear ownership boundaries

Within each milestone, subfolders should allow parallel development with minimal overlap.
Developers should work by module ownership, not by editing the same orchestration files repeatedly.

### Stable naming

Generated assets must use consistent naming conventions.
No ambiguous filenames such as final.csv, new_final.json, or test2_output.txt.

### No hidden path logic

All paths should be derived centrally through shared path utilities rather than scattered hardcoded strings.

Reports must reflect implementation

Milestone reports and FS sections must align with the actual structure and terminology of the repository.

### Keep modules cohesive

Each submodule should do one coherent job.
Do not mix ingestion, cleaning, evaluation, and export logic into the same file unless the scope is tiny and clearly justified.

### Branching Convention

The repository will maintain milestone-oriented progression using dedicated milestone branches.

Expected long-lived branches:

- main
- milestone-1
- milestone-2
- milestone-3

Expected short-lived feature branches:

- feat/ms1-profiling
- feat/ms1-cleaning
- feat/ms1-dataset-build
- feat/ms2-rnn
- feat/ms2-transformer
- feat/ms2-metrics
- feat/ms3-indexing
- feat/ms3-cache
- feat/ms3-prompting

This preserves milestone sequencing while still enabling parallel development inside each milestone.

### Consequences

Positive

- milestone boundaries are explicit
- shared infrastructure is centralized
- parallel work is easier
- later milestones can build on earlier outputs cleanly
- the repo structure aligns with reporting and delivery
- execution becomes standardized

Trade-offs

- the team must be disciplined about what belongs in common
- developers must avoid slipping into ad hoc scripts
- structure must be maintained actively rather than assumed

## Final Rule

If a new file does not clearly belong somewhere in this structure, the default action is not to invent a vague folder.
Instead, the team must decide whether it is:

1. shared infrastructure,
2. milestone-specific logic,
3. data,
4. experiment output, or
5. documentation.
