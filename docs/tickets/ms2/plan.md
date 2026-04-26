# Milestone 2 Implementation Plan

This file is the source-of-truth implementation plan for Milestone 2 (the "RNN vs. Transformer, both from scratch" comparison). Every ticket below is the dev-facing translation of a section in the design freeze ADR at `docs/fs/ms2/ADR.md`. Read the ADR first; this plan is intentionally thin on architectural justification and thick on contract/scope/acceptance.

Conventions match MS1: each ticket has Description, Acceptance Criteria, Produces Artifacts, Blocking Class, and Notes (Dependencies + Story Points). Ticket IDs follow `MS2-<STREAM>-<INDEX>` per `docs/tickets/README.md`.

The ADR itself is `MS2-DESIGN-01` (status: design freeze, awaiting team review per ADR §7). All tickets below assume the ADR is the binding contract; if any acceptance criterion below disagrees with the ADR, the ADR wins.

---

## Workload Summary

- 21 tickets, 67 story points total; 1 ticket / 2 story points completed (`MS2-INFRA-01`).
- Streams: `INFRA` (4), `DATA` (3), `TRAIN` (2), `EVAL` (2), `MODEL-A` (3), `MODEL-B` (3), `INFER` (1), `ABLATE` (1), `COMPARE` (1), `REPORT` (1).
- Parameter / compute budgets are fixed by ADR §2.14, §3.9, §4.1: ~2.7M params for Model A, ~3.8M for Model B, ~75 min wall-clock per training run, 3 seeds (`{13, 42, 91}`) per model.

Per-developer assignment will be performed in `alice.md`, `bob.md`, `charly.md` once this plan is reviewed and frozen. The recommended execution order is in the final section of this document.

---

## MS2-INFRA-01 [DONE]: Repository Paths, Mixed-Precision Policy, and Reproducibility Helpers

### Description

Extend the MS1 path/output infrastructure (`src/common/paths.py`, `make_ms1_output_filename`) with MS2-specific resolvers and helpers. Anchor every artifact MS2 produces to a deterministic, named path so that downstream tickets do not invent paths ad hoc.

The ticket also lands the global numerical-policy and reproducibility surface that the ADR mandates in §1.5 (`mixed_float16` global policy with `LossScaleOptimizer`-friendly hooks; loss must compute in float32) and §1.9 (`tf.keras.utils.set_random_seed`, `tf.config.experimental.enable_op_determinism`, fixed-seed tokenizer training). These are framework-level concerns that have to be set _exactly once at process start_ — putting them in this ticket prevents every downstream ticket from re-implementing them.

The MS2 directory layout adds, at minimum: `data/processed/ms2/` (tokenizer, char vocab, TFRecord cache, length analysis), `experiments/ms2/<model>/<seed>/` (per-run training curves, checkpoints, eval JSON), and `docs/fs/ms2/` (artifact map and contracts, alongside the existing ADR). All MS2 modules must consume `resolve_ms2_paths(...)` and `make_ms2_output_filename(...)` analogues to MS1.

### Acceptance Criteria

- `src/common/paths.py` (or a new `src/common/paths_ms2.py` co-located with the MS1 helpers) exposes:
  - `MS2Paths` dataclass with fields for processed data root, experiment root, tokenizer path, char vocab path, TFRecord shard dir, run-output dir, and report dir.
  - `resolve_ms2_paths(..., create_dirs: bool = True)` that auto-creates writable dirs (mirroring MS1 behavior).
  - `make_ms2_output_filename(stage, name, version, ext)` enforcing the snake*case `ms2*<stage>\_<name>\_v###.<ext>` contract.
- A `src/ms2/runtime.py` module exposes `configure_runtime(seed: int, mixed_precision: bool = True, deterministic: bool = True)` that:
  - Sets the global Keras mixed-precision policy to `mixed_float16` when enabled (ADR §1.5).
  - Calls `tf.keras.utils.set_random_seed(seed)` and `tf.config.experimental.enable_op_determinism()` for deterministic seeded runs (ADR §1.9).
  - Logs an informational summary so seeded runs are reproducible from logs alone.
- No hardcoded relative paths to `data/processed/ms2/` or `experiments/ms2/` exist in MS2 modules.
- A path resolution example output is captured for the artifact map (analogous to MS1's `ms1-path-resolution-example.txt`).
- `pyproject.toml` package list updated so `src.ms2`, `src.ms2.<subpkg>` are properly packaged (per AGENTS.md note about the explicit setuptools list).

### Produces Artifacts

- `src/common/paths.py` updates (or `src/common/paths_ms2.py`)
- `src/ms2/runtime.py`
- `docs/fs/ms2/ms2-artifact-map.md`
- Path resolution example snapshot
- Updated `pyproject.toml` package boundaries

### Blocking Class

Hard blocker.

### Notes

- Dependencies: MS1 infra (already done — `MS1-INFRA-01`, `MS1-INFRA-02`).
- Story points: 2.

---

## MS2-INFRA-02: Core MS2 Schemas and Run Configuration

### Description

Define the importable Python data structures used throughout MS2. Unlike MS1 (where the heavy schema work was raw transcript and QA records), MS2's schema layer is dominated by **training-side** and **result-side** records: configuration objects that pin a run, batch tensor specs, per-step metric records, and per-run summary records that the comparison ticket will load.

The minimum surface, all in `src/ms2/schemas.py` (or `src/common/schemas.py` extensions):

- `RunConfig`: the frozen-at-launch description of a training run — model id (`A` or `B`), seed, length caps (`L_q`, `L_c`, `L_enc`, `L_dec` from ADR §1.3), batch-construction settings (bucket boundaries, target tokens-per-batch from ADR §1.4), LR schedule choice (cosine-with-warmup vs. Noam, see ADR §2.12 / §3.7), wall-clock budget (75 min default), label-smoothing `ε_ls = 0.1` (ADR §1.7), gradient-clip norm `1.0` (ADR §2.12 / §3.7).
- `BatchSpec`: declarative description of the encoder/decoder tensor shapes, padding ID (= 0, ADR §1.1), char-matrix shape `(B, L_enc, L_char_max)` with `L_char_max = 16` (ADR §2.6).
- `RunSummary`: dev/test EM / Token-F1 / char edit distance / BLEU-1 (ADR §1.8) plus parameter count, peak GPU memory, train wall-clock, mean inference time / example. Designed to feed directly into the §4.3 comparison table.
- `AblationConfig`: lightweight extension over `RunConfig` describing which switch is flipped (one of: `no_film`, `mean_merge`, `plain_branch3`, `sinusoidal_pe`, `no_pe`, `shared_layers`).

All schemas should have JSON-serializable `to_dict` / `from_dict` round-trips so that on-disk artifacts are inspection-friendly and the comparison ticket can load them without hand-rolling a parser.

### Acceptance Criteria

- `src/ms2/schemas.py` defines `RunConfig`, `BatchSpec`, `RunSummary`, `AblationConfig`, with type annotations and docstrings that cite the ADR section each field comes from.
- Each schema has at least one constructed `EXAMPLE_*` constant (mirrors MS1's `EXAMPLE` test contract — see `tests/common/test_schemas.py`).
- A test file `tests/ms2/test_schemas.py` round-trips each schema through JSON and asserts equality.
- `RunConfig.frozen_dict()` is hashable, so two configs that disagree on any binding field hash differently.
- A small field-dictionary table is captured in `docs/fs/ms2/ms2-schema-contract.md` (mirrors `docs/fs/ms1-schema-contract.md`).

### Produces Artifacts

- `src/ms2/schemas.py`
- `docs/fs/ms2/ms2-schema-contract.md`
- `tests/ms2/test_schemas.py`
- JSON examples for each schema, stored under `docs/fs/artifacts/ms2/`.

### Blocking Class

Hard blocker.

### Notes

- Dependencies: `MS2-INFRA-01` [DONE].
- Story points: 2.

---

## MS2-INFRA-03: MS2 CLI Interface

### Description

Define and implement the MS2 CLI entrypoints, mirroring the MS1 pattern at `src/cli/ms1.py`. Per the project's execution model (`docs/fs/overall.md` §6), all MS2 functionality must be reachable via `python -m src.cli.ms2 <command>`. CLI files orchestrate modules; they must not contain business logic.

The set of commands has to be sufficient for the entire milestone — preparation, training, inference, evaluation, ablation, comparison, end-to-end — but no command should expose more than 4–5 user-facing flags. A `RunConfig` JSON file (per `MS2-INFRA-02`) is the canonical mechanism for parameter-heavy invocations.

Required commands:

- `prep-data`: tokenizer training + char vocab + TFRecord cache build (orchestrates `MS2-DATA-02`, `MS2-DATA-03`).
- `analyze-lengths`: length distribution analysis from MS1 output (`MS2-DATA-01`).
- `train`: trains one model on one seed; takes `--model {a,b}` and `--seed {13,42,91}` and `--config <path>`.
- `infer`: runs greedy and/or beam decoding on the dev/test split for a trained run (`MS2-INFER-01`).
- `evaluate`: computes EM / F1 / char-edit-dist / BLEU-1 against references (`MS2-EVAL-01`).
- `evaluate-protocol`: runs the noise battery, leave-2-videos-out, long-dependency, and difficulty-bucket protocols (`MS2-EVAL-02`).
- `ablate`: runs an ablation variant; `--variant {no_film, mean_merge, plain_branch3, sinusoidal_pe, no_pe, shared_layers}` (`MS2-ABLATE-01`).
- `compare`: builds the headline comparison table and diagnostic plots from completed run outputs (`MS2-COMPARE-01`).
- `run-all`: end-to-end orchestration for the whole milestone (`MS2-INFRA-04`).

CLI output format follows the MS1 pattern (`[<command>] status=...`) so the CLI smoke-test pattern at `tests/cli/test_ms1_cli.py` can be reused.

### Acceptance Criteria

- `src/cli/ms2.py` exists and is registered as an installed script in `pyproject.toml` (`[project.scripts] ms2 = "src.cli.ms2:main"`).
- All commands listed above run with `--help` and emit usage text.
- CLI output pattern matches `[<command>] status=...` (so the existing CLI smoke pattern works).
- `run-all` executes the right sub-commands in dependency order (asserted by a CLI test analogous to `tests/cli/test_ms1_cli.py`).
- A CLI help-output snapshot is captured.

### Produces Artifacts

- `src/cli/ms2.py`
- CLI help output snapshot
- Example execution log
- `docs/fs/ms2/ms2-cli-contract.md`
- `tests/cli/test_ms2_cli.py`

### Blocking Class

Hard blocker.

### Notes

- Dependencies: `MS2-INFRA-01` [DONE], `MS2-INFRA-02`.
- Story points: 2.

---

## MS2-INFRA-04: End-to-End MS2 Pipeline (`run-all`)

### Description

The integration ticket. Wire all stage modules into a single `run-all` command that, from a clean repo checkout, executes the full milestone: data prep → training (both models × 3 seeds) → inference → evaluation → ablations → comparison → report scaffold. This mirrors `MS1-INFRA-05`.

The pipeline must be **resumable**: a stage that finds its declared output artifact already present must skip itself unless `--force` is passed. This keeps the wall-clock cost of incremental development down (each model training is ~75 min — ADR §3.7, §2.12 — so re-running everything from scratch is expensive).

The pipeline orchestrator is also the right place to enforce the "matched compute budget" contract from ADR §4.1 (~75 min per training run): the orchestrator records the wall-clock per stage and warns loudly if any model run materially exceeds budget.

### Acceptance Criteria

- `python -m src.cli.ms2 run-all` executes the complete MS2 pipeline.
- Outputs land in canonical directories from `MS2-INFRA-01` [DONE].
- Pipeline is idempotent: re-running with all artifacts present is a fast no-op.
- `--force` flag triggers re-execution from a chosen stage onwards.
- A wall-clock summary is emitted per stage; deviations from the ADR-declared budgets are flagged.
- Execution log captured under `experiments/ms2/run_all_log_v###.txt`.
- An integration checklist (mirrors MS1) is produced.
- A "known limitations" note is produced (e.g., compute-bound seeds, ablation 3 of Model B optional).

### Produces Artifacts

- Full pipeline execution log
- MS2 artifact manifest
- Integration checklist
- Known limitations note

### Blocking Class

Soft blocker (downstream tickets can run individual sub-commands; `run-all` is for reproducibility).

### Notes

- Dependencies: `MS2-INFRA-03` and every implementation ticket below (`MS2-DATA-*`, `MS2-MODEL-A-*`, `MS2-MODEL-B-*`, `MS2-TRAIN-*`, `MS2-EVAL-*`, `MS2-INFER-01`, `MS2-ABLATE-01`, `MS2-COMPARE-01`).
- Story points: 2.

---

## MS2-DATA-01: Length Distribution Analysis and Length-Cap Freeze

### Description

ADR §1.3 explicitly defers the final length caps (`L_q`, `L_c`, `L_enc`, `L_dec`) to this ticket — the placeholder values (32 / 384 / 420 / 64) are to be confirmed against the actual percentiles of the cleaned MS1 corpus. The MS1→MS2 handoff (`docs/fs/artifacts/ms1/ms1-ms2-handoff-note.md`) gives us the canonical export at `data/processed/ms1/ms1_dataset_processed_v001.jsonl`; this ticket consumes that.

For each of the four length axes, compute the empirical CDF and propose a cap at the **95th percentile**, **99th percentile**, and **maximum**. Choose a cap that covers ≥ 99 % of examples without truncation while keeping `L_enc ≤ 420` to respect the bucket-boundary contract in ADR §1.4. If 99 % coverage requires a larger cap, document the trade-off and update the ADR placeholder via a small ADR-amendment note (do _not_ silently bump the bucket boundaries — that has knock-on memory effects).

The deliverable is not just numbers but a frozen, importable `LENGTH_CAPS` constant so every downstream ticket reads the same values. This is the equivalent of MS1's `MS1-ANALYSIS-01` corpus-distribution ticket but scoped tightly to the length-cap decision.

### Acceptance Criteria

- Empirical distribution computed for: question token length, context token length (BPE-tokenized — note tokenizer doesn't exist yet, so this ticket can use whitespace + punctuation segmentation as a proxy with a clear caveat in the report; final caps are re-validated post-`MS2-DATA-02` with a 1-line correction if they shift), encoder-input length (`L_q + L_c + 4` per ADR §1.3), decoder length.
- Histograms saved per axis.
- 95th / 99th / max percentiles reported in a table.
- Final caps committed as a Python constant (`LENGTH_CAPS` in `src/ms2/data/length_caps.py`) and linked from `RunConfig`.
- If any cap differs from the ADR placeholder by ≥ 10 %, a short ADR-amendment note is added.
- ≥ 99 % example coverage at the chosen `L_enc` cap is verified.

### Produces Artifacts

- `experiments/ms2/length_distribution_v001.json`
- 4 histogram PNGs (`docs/fs/artifacts/ms2/length_*.png`)
- `src/ms2/data/length_caps.py`
- Short observation note (markdown)
- Optional ADR amendment

### Blocking Class

Hard blocker (model code reads `LENGTH_CAPS`).

### Notes

- Dependencies: `MS2-INFRA-01` [DONE], `MS2-INFRA-02`. Reads MS1 handoff artifact.
- Story points: 2.

---

## MS2-DATA-02: BPE-4k Tokenizer and Character Vocabulary

### Description

Train and freeze the two text-discretization assets every model uses. ADR §1.1 specifies BPE via SentencePiece, `V = 4096`, character coverage 1.0, special token IDs 0–4 (`<pad>`, `<unk>`, `<bos>`, `<eos>`, `<sep>`), trained on transcript text **only** (the ADR is explicit that QA pairs must not leak into the tokenizer corpus). The output artifact path is fixed: `data/processed/ms2/ms2_tokenizer_bpe_4k_v001.model`.

ADR §2.6 separately requires a character vocabulary for Branch 3 of Model A: roughly 150 entries covering Arabic + Latin + digits + punctuation + a `<pad>` character. The vocabulary must be deterministic (sort by codepoint) so that re-running training does not shuffle IDs and break checkpoints.

Tokenizer training must be deterministic per ADR §1.9 — single-threaded, fixed seed.

This ticket also writes a thin wrapper module `src/ms2/data/tokenizer.py` that exposes `encode(text: str) -> list[int]`, `decode(ids: list[int]) -> str`, and `encode_chars(token_str: str, max_chars: int = 16) -> list[int]`. The character encoding right-truncates to `L_char_max = 16` per ADR §2.6 and pads with the character `<pad>`.

### Acceptance Criteria

- BPE tokenizer trained with vocab size exactly 4096; special IDs 0–4 verified by direct lookup.
- Tokenizer training is deterministic — two consecutive trainings with identical inputs yield identical vocabularies (byte-equal `.model` files, or at minimum identical id-to-piece mappings).
- Character vocabulary covers the union of characters appearing in the cleaned corpus; verified by exhaustive scan.
- `encode` round-trips: a sample of 100 cleaned strings, `decode(encode(s))`, normalizes to the same string after Arabic post-normalization (ADR §1.8) — full lossless round-trip is not guaranteed by BPE, but post-norm equality is the right contract.
- Character coverage of 1.0 verified — no `<unk>` triggered on the training corpus characters.
- Tokenizer wrapper exposes the three documented functions and is type-checked via tests.
- Tokenizer trained on transcript text only (QA pairs excluded) — verified by a test reading the training input list.

### Produces Artifacts

- `data/processed/ms2/ms2_tokenizer_bpe_4k_v001.model`
- `data/processed/ms2/ms2_char_vocab_v001.json`
- `src/ms2/data/tokenizer.py`
- `tests/ms2/test_tokenizer.py`
- A small "tokenizer card" doc capturing vocab size, special tokens, training corpus, training command (mirrors the spirit of `docs/fs/ms1-schema-contract.md`).

### Blocking Class

Hard blocker.

### Notes

- Dependencies: `MS2-INFRA-01` [DONE], `MS2-DATA-01`. Reads `data/processed/ms1/ms1_dataset_processed_v001.jsonl`.
- Story points: 3.

---

## MS2-DATA-03: tf.data Input Pipeline (Sequence Formatting, Context Windowing, Bucketing, Char Matrix, TFRecord Cache)

### Description

Build the training-time data pipeline. This is a large, single-owner ticket because the pieces are tightly coupled and breaking them apart would force every consumer to re-stitch.

The pipeline must implement, per ADR:

1. **Sequence formatting** (§1.2): encoder input = `[<bos>] question_tokens [<sep>] context_tokens [<eos>]`; decoder input = `[<bos>] answer_tokens`; decoder target = `answer_tokens [<eos>]`. Loss mask is true where the target is not `<pad>`.
2. **Context windowing** (§1.3): training-time uses a `L_c`-length window centered on the gold answer span with random jitter `±20%` of `L_c`. Inference-time uses sliding windows with stride `L_c / 2` over the full transcript when the transcript exceeds `L_c`.
3. **Per-token character matrix** (§2.6): for the joint encoder input, build a `(L_enc, L_char_max=16)` integer tensor where row `i` holds the right-truncated character IDs of BPE token `i`, padded with the character `<pad>`. This is needed only for Model A's Branch 3 — Model B does not consume it, so the pipeline must support **two output schemas**: one for Model A (with char matrix), one for Model B (without). A `--target-model {a,b}` flag on the prep stage selects.
4. **Bucketing & batching** (§1.4): `tf.data.Dataset.bucket_by_sequence_length` on `L_enc` with bucket boundaries `[128, 192, 256, 320, 420]`; per-bucket batch size scaled inversely with length so total tokens per batch ≈ 16,384; shuffle buffer 4096; cache to TFRecord to avoid re-tokenization across epochs.
5. **Padding-mask construction** is the consumer's job (the model code derives it from `input_ids != 0`), but the pipeline must guarantee that `<pad>` is always ID 0 (verified at write time).

The same pipeline serves training, dev, and test, but only training uses the random jitter — dev/test use a deterministic centered window.

### Acceptance Criteria

- Encoder/decoder/target tensors match the formatting in ADR §1.2 byte-for-byte (verified by a test that hand-constructs an example).
- Random jitter on training context windows verified statistically (mean offset ≈ 0, std within `0.2 · L_c`).
- Inference-time sliding windows with stride `L_c / 2` verified for a transcript longer than `L_c`.
- Char matrix (Model A schema) has shape `(L_enc, 16)` and right-truncates correctly; padding char ID consistent.
- Bucketing produces the configured boundaries; mean tokens-per-batch within ±10 % of 16,384.
- TFRecord cache is created on first run and reused on second run (timing test: second pass through epoch 0 is ≥ 5× faster than first pass).
- Two pipeline schemas (Model A vs. Model B) selectable; both validated by tests.
- Padding ID is always 0 in produced tensors.

### Produces Artifacts

- `src/ms2/data/pipeline.py`
- `data/processed/ms2/ms2_tfrecords_train_v001/` (and `_dev_`, `_test_` siblings, per Model A and Model B variants)
- `tests/ms2/test_pipeline.py`
- A small "pipeline contract" doc documenting input/output schemas

### Blocking Class

Hard blocker.

### Notes

- Dependencies: `MS2-DATA-01`, `MS2-DATA-02`.
- Story points: 4.

---

## MS2-TRAIN-01: Training Utilities (Optimizer, LR Schedules, Loss, Gradient Clipping, Loop)

### Description

Build the training-side utilities both models share, plus the two model-specific LR schedules. The common surface (ADR §1.5–§1.7, §4.1) is:

- AdamW optimizer with `β_1 = 0.9, β_2 = 0.98, ε = 1e-9, weight_decay = 0.01`, wrapped in `LossScaleOptimizer` (mixed precision, ADR §1.5).
- Weight decay applied to all weights _except_ embeddings, LayerNorm parameters, and biases (ADR §2.12 / §3.7) — implement via parameter groupings or a `decay_var_filter` callable.
- Categorical cross-entropy with label smoothing `ε_ls = 0.1`, computed in float32 for numerical stability, masked on `<pad>` decoder-target positions, mean-reduced over unmasked tokens (ADR §1.7).
- Global-norm gradient clipping at `1.0` (ADR §2.12 / §3.7).
- Teacher-forcing ratio fixed at 1.0 (scheduled sampling is an ablation, not a default — ADR §2.12 last bullet).

Model-specific schedules:

- **Model A** — cosine decay with warmup. Peak LR `3e-4`, warmup over the first 5 % of training steps, decay to `1e-5` over the remaining steps (ADR §2.12).
- **Model B** — Noam: `lr(step) = d_model^(-0.5) · min(step^(-0.5), step · warmup^(-1.5))`, warmup = 1000 steps (ADR §3.7).

The ticket also implements a `train_one_run(model, dataset_train, dataset_dev, run_config) -> RunSummary` driver that runs to wall-clock budget (75 min target), checkpoints best-on-dev and last, emits per-step training loss + per-epoch dev metrics, and writes a `RunSummary` JSON.

### Acceptance Criteria

- `src/ms2/training/optimizer.py` exposes `build_adamw(...)` with the parameter-group filtering described.
- `src/ms2/training/schedules.py` implements `CosineWithWarmup` and `Noam`, both as `tf.keras.optimizers.schedules.LearningRateSchedule` subclasses; LR-vs-step curves match the equations within fp32 tolerance.
- `src/ms2/training/loss.py` exposes `label_smoothed_cross_entropy(logits, targets, mask, smoothing=0.1)` computed in float32.
- Loss is provably zero on a hand-crafted batch where logits are one-hot at the targets and there's no smoothing (with smoothing=0.1, expected value ≈ entropy of the smoothed target).
- Mask correctness: padded positions contribute zero gradient (verified by a test that flips a padded target and confirms loss is unchanged).
- `train_one_run(...)` enforces the wall-clock budget; emits the expected artifacts under `experiments/ms2/<model>/<seed>/`.
- Mixed-precision policy is honored: all model-internal float ops in float16, loss in float32, optimizer wrapped with `LossScaleOptimizer`.

### Produces Artifacts

- `src/ms2/training/{optimizer,schedules,loss,loop}.py`
- `tests/ms2/test_training_utils.py`
- LR-schedule visualization PNGs (one per schedule, for sanity)

### Blocking Class

Hard blocker.

### Notes

- Dependencies: `MS2-INFRA-01` [DONE], `MS2-INFRA-02`. Independent of model implementations (uses dummy models in tests).
- Story points: 3.

---

## MS2-EVAL-01: Metrics Implementation and Arabic Post-Normalization

### Description

Implement the four evaluation metrics specified in ADR §1.8, plus the unified Arabic post-normalization step that all metrics consume. The post-normalization is binding for both models — the ADR is explicit that "same normalization for both models", so the implementation must be a single function used everywhere.

Post-normalization (ADR §1.8):

- Alif/hamza unification (`أ`, `إ`, `آ` → `ا`).
- Ya / alif-maqsura unification (`ى` → `ي`).
- Diacritic stripping (Tashkeel marks).

Metrics:

- **Exact Match (EM)**: equality after post-normalization.
- **Token-F1**: SQuAD-style F1 over post-normalized whitespace-tokenized bag-of-words.
- **Character edit distance (normalized)**: Levenshtein distance / `max(len(pred), len(ref))`.
- **BLEU-1**: unigram BLEU as a soft secondary signal.

The metrics module must be tested against hand-crafted reference cases (e.g., F1 on `pred="الذهب الأصفر" / ref="الذهب الأصفر"` after normalization should be 1.0; on disjoint bags should be 0.0; on partial overlap should match the closed-form F1 expression).

The post-normalization pipeline is reused by Branch 3 / Model A's character vocabulary scan implicitly; this ticket therefore also stress-tests that the same normalization produces the same character set the char vocab covers.

### Acceptance Criteria

- `src/ms2/metrics/normalize.py` exposes `arabic_post_normalize(text: str) -> str`.
- Hamza/alif/ya/maqsura/diacritic rules verified by a comprehensive table-driven test (≥ 20 cases).
- `src/ms2/metrics/scoring.py` exposes `exact_match`, `token_f1`, `char_edit_distance_normalized`, `bleu1`, all batch-friendly.
- Metric correctness asserted on hand-crafted cases including: exact match, zero overlap, partial overlap, prediction-shorter-than-reference, prediction-longer-than-reference.
- Aggregation utility `aggregate_metrics(predictions, references) -> RunSummary`-compatible dict.
- All metrics are computed _after_ post-normalization on both prediction and reference.

### Produces Artifacts

- `src/ms2/metrics/normalize.py`
- `src/ms2/metrics/scoring.py`
- `tests/ms2/test_metrics.py`
- Short "metric contract" doc summarizing definitions and edge cases.

### Blocking Class

Hard blocker (`MS2-EVAL-02`, `MS2-COMPARE-01`, and the training loop's dev evaluation all depend on this).

### Notes

- Dependencies: `MS2-INFRA-01` [DONE].
- Story points: 3.

---

## MS2-MODEL-A-01: Model A — Embedding, 3 Encoder Branches, Char-CNN

### Description

Implement the bottom of Model A's encoder stack: the BPE token embedding (ADR §2.3), Branch 1 (question Bi-LSTM + structured pooling, ADR §2.4), Branch 2 (context 2× Bi-GRU, ADR §2.5), and Branch 3 (joint encoder with char-CNN + BPE concat + Bi-LSTM, ADR §2.6).

Implementation must use the **Subclassing API** per ADR §2.16. Implementation notes from ADR §2.16 are binding:

- Char-CNN: reshape `(B, L_enc, L_char_max, d_char) → (B·L_enc, L_char_max, d_char)` before Conv1D, three parallel Conv1D with kernel sizes `{2, 3, 4}` and `filters = d_charcnn / 3` each, GlobalMaxPool1D over chars per token, concatenate, reshape back.
- LSTMs/GRUs: keep cuDNN-friendly defaults (`activation='tanh'`, `recurrent_activation='sigmoid'`, no projection, default kernel init).
- Bi-LSTM / Bi-GRU via `tf.keras.layers.Bidirectional` — concatenation merge mode (default).
- Token embedding initialized with `RandomNormal(stddev = d_tok ** -0.5)`.
- Branch 1 structured pooling (ADR §2.4): `scores_i = w_p · tanh(W_p · b1_seq_i + b_p)`, `α = softmax_i(scores_i)`, output `Σ α_i b1_seq_i`. The softmax must be **masked over `<pad>` positions before normalization** (ADR §2.16 last bullet). `d_p = 64`.
- Branches 1 and 3 share the BPE embedding matrix with the decoder (tied, ADR §2.3 + §2.11).

This ticket _does not_ tie the three branches into a single model — the gated merge / refinement / FiLM / decoder live in `MS2-MODEL-A-02` and `MS2-MODEL-A-03`.

### Acceptance Criteria

- `src/ms2/models/rnn/embedding.py` implements the shared BPE embedding (with init per ADR §2.3) and the character embedding `(V_char, d_char=32)`.
- `src/ms2/models/rnn/branch1.py` produces `b1_pooled : (B, 2·d_b1=192)` from `(B, L_q)` ids; structured-pool softmax respects the padding mask (verified by test: padded positions get 0 weight).
- `src/ms2/models/rnn/branch2.py` produces `b2_seq : (B, L_c, 2·d_b2=192)` from `(B, L_c)` ids via two stacked Bi-GRUs.
- `src/ms2/models/rnn/branch3.py` produces `b3_seq : (B, L_enc, 2·d_b3=192)` from `(B, L_enc)` ids and `(B, L_enc, 16)` char ids.
- Char-CNN matches ADR §2.6: 3 kernels in `{2,3,4}`, GlobalMaxPool over chars, concatenated to `(B, L_enc, d_charcnn=64)`.
- All hyperparameter values match ADR §2.2 exactly.
- A shape-test asserts each branch's output shape on a `(B=2, L_q=32, L_c=384, L_enc=420)` mini-batch.
- Branches expose padding masks downstream (mask propagation verified).

### Produces Artifacts

- `src/ms2/models/rnn/{embedding, branch1, branch2, branch3, char_cnn}.py`
- `tests/ms2/test_model_a_branches.py`
- A small architecture diagram for the three branches (ASCII or PNG, in `docs/fs/artifacts/ms2/`)

### Blocking Class

Hard blocker (gated merge depends on all three branch outputs).

### Notes

- Dependencies: `MS2-INFRA-01` [DONE], `MS2-INFRA-02`, `MS2-DATA-02`. Reads no actual data — pure layer construction with shape tests.
- Story points: 4.

---

## MS2-MODEL-A-02: Model A — Alignment, Gated Merge (Per-Position MoE), Refinement Bi-GRU, FiLM Generator

### Description

Implement the middle of Model A: the alignment step (ADR §2.7), the gated merge block which is the conceptual centerpiece of Model A (ADR §2.8), the refinement Bi-GRU (ADR §2.9), and the FiLM generator (ADR §2.10).

Key implementation contracts:

- **Alignment (§2.7)**: `b1_pooled` is **tiled** along the context axis to `(B, L_c, 192)`; `b2_seq` is already `(B, L_c, 192)`; `b3_seq` is **sliced to context positions only** via offset `L_q + 2` (after `<bos>` and `<sep>`) for length `L_c`, computed per-example using the runtime question length. The slicing must use `tf.gather` (or equivalent) parameterized by per-example offset — a static slice will be wrong on right-padded questions.
- **Gated merge (§2.8)**: implement Steps 1–4 exactly. The defense in §2.8 ("attention computes pairwise compatibility scores between positions; this is mixture-of-experts gating with three experts") is part of the architecture's identity. Concretely: `W_1, W_2, W_3` project each branch to `d_fuse = 192`; `concat ∈ ℝ^{3·d_fuse}`; `hidden = GELU(W_g · concat + b_g) ∈ ℝ^{d_fuse}`; `logits = w_out · hidden + b_out ∈ ℝ^3`; `gates = softmax(logits)`; `fused = gates[0]·h1 + gates[1]·h2 + gates[2]·h3`; `LayerNorm(Dropout(fused, 0.3))`.
- **Refinement Bi-GRU (§2.9)**: single Bi-GRU layer with `d_refine = 96` per direction, `return_sequences=True, return_state=True`. The forward and backward final states are concatenated to form `encoder_summary : (B, d_enc_out=192)`.
- **FiLM generator (§2.10)** — Option 1, Static Global. **Identity initialization is critical**: `W_f2` initialized with `stddev=1e-3`; bias split as `(b_γ, b_β)` with `b_γ = 1` (vector of ones, length `d_dec=256`) and `b_β = 0`. At step 0, `γ ≈ 1, β ≈ 0`, FiLM is identity. The ADR explicitly calls this convention "project-wide" — re-use it for any future FiLM block.

The ticket also exposes the **gates tensor** as a model output (auxiliary), so the diagnostic visualization in `MS2-COMPARE-01` (item 5: gated-merge weights) can plot it without extra forward passes.

### Acceptance Criteria

- `src/ms2/models/rnn/align.py` produces three aligned tensors `(B, L_c, 192)` with the slice/tile contracts above; per-example offset slicing tested on a synthetic batch with non-uniform question lengths.
- `src/ms2/models/rnn/gated_merge.py` matches §2.8 byte-for-byte; gates sum to 1 along the branch axis (verified numerically).
- `src/ms2/models/rnn/refinement.py` produces `(encoder_output, encoder_summary)` of correct shapes.
- `src/ms2/models/rnn/film_generator.py` produces `(γ, β)` of shape `(B, d_dec=256)` each.
- **Identity init verified**: at construction time, before any training, with random `z`, `γ` is within `1 ± 0.01` and `β` is within `0 ± 0.01` (for `stddev=1e-3` init this is statistically robust over 100 random `z` vectors).
- Shape-test on the assembled middle stack (alignment → gated_merge → refinement → FiLM gen) on a `(B=2, L_c=384)` batch.
- Gates tensor exposed as auxiliary model output for visualization.

### Produces Artifacts

- `src/ms2/models/rnn/{align, gated_merge, refinement, film_generator}.py`
- `tests/ms2/test_model_a_middle.py`
- Architecture diagram including alignment / merge / FiLM data flow

### Blocking Class

Hard blocker.

### Notes

- Dependencies: `MS2-MODEL-A-01`.
- Story points: 4.

---

## MS2-MODEL-A-03: Model A — FiLM-Conditioned LSTM Decoder, Tied Output Projection, Full Assembly

### Description

Implement the top of Model A and the full-model glue. This ticket also performs the parameter-budget audit specified in ADR §2.14.

Contracts from ADR §2.11:

- 2-layer unidirectional LSTM, `d_dec = 256` per layer.
- Per-layer initial state: `h_0^{(l)} = tanh(W_h^{(l)} · z + b_h^{(l)})`, `c_0^{(l)} = tanh(W_c^{(l)} · z + b_c^{(l)})`. Four projections total (separate per layer for `h` and `c`).
- FiLM applied at the **output of each layer**: `h_l = γ[:, None, :] ⊙ h_l + β[:, None, :]`. The same `(γ, β)` from the FiLM generator is broadcast across time and reused at both layers (ADR §2.11 makes this explicit).
- Output projection: tied to the BPE embedding via `logits = h2_proj · E^T + b_out`, where `h2_proj = W_proj · h2` (because `d_dec=256 ≠ d_tok=128`, the linear projection is mandatory; ADR §2.11 last paragraph). `W_proj` is **not** tied; it's a standalone learned matrix `(d_tok × d_dec)`. `b_out` is a learned bias of shape `(V,)`.
- Tied embedding implementation must follow ADR §2.16: the output projection holds a **reference** to the embedding's `embeddings` variable, not a copy. `model.trainable_variables` must not double-count the embedding.

The full assembly stitches:

```
inputs → branches (-A-01) → align/merge/refinement/FiLM-gen (-A-02) → decoder → tied output → logits
```

Parameter audit (ADR §2.14): produce a printed parameter breakdown matching the table; total within ±10 % of the ~2.7 M target. If grossly off, document why.

### Acceptance Criteria

- `src/ms2/models/rnn/decoder.py` implements the 2-layer LSTM + per-layer FiLM modulation + tied output projection.
- Initial-state projections produce correct shapes; tested.
- FiLM modulation broadcasts correctly across the time axis; tested with a hand-crafted `(γ=1, β=0)` example reducing to identity.
- Output projection is **tied**: `model.trainable_variables` does not double-count the embedding (verified by checking the count and identity of variables).
- `src/ms2/models/rnn/model_a.py` exposes a `ModelA(tf.keras.Model)` subclass with `call(inputs, training)` taking `(question_ids, context_ids, joint_ids, char_ids, decoder_inputs)` and returning `logits` plus auxiliary `(gates, gamma, beta, encoder_summary)` for diagnostics.
- Forward-pass shape test on a `(B=2)` batch end-to-end.
- Parameter audit: printed table breakdown matches ADR §2.14 categories; total within `[2.4M, 3.0M]`.
- A loss-decreases sanity check: 200 training steps on a tiny batch overfits training loss to near zero (Adam, lr=1e-3, no smoothing).

### Produces Artifacts

- `src/ms2/models/rnn/{decoder, model_a}.py`
- `tests/ms2/test_model_a_full.py`
- Parameter audit JSON + printed breakdown
- "Model A architecture card" doc summarizing layer-by-layer shapes (mirrors a model card)

### Blocking Class

Hard blocker.

### Notes

- Dependencies: `MS2-MODEL-A-01`, `MS2-MODEL-A-02`.
- Story points: 4.

---

## MS2-MODEL-B-01: Model B — RoPE Implementation with Unit Tests

### Description

Implement Rotary Positional Embeddings as a standalone, well-tested layer. ADR §3.4 is the binding mathematical contract — read it carefully, especially §3.4.2 (the rotation), §3.4.3 (the relative-position property), §3.4.4 (efficient implementation), and §3.4.5 (the application table).

Contracts:

- For `d = d_head = 48`, split each query/key vector into 24 consecutive 2D pairs.
- Frequencies `θ_i = θ_base ^ (-2(i-1)/d) = 10000 ^ (-2(i-1)/48)`, `i ∈ {1, ..., 24}`. This gives `θ_1 = 1.0` and `θ_{24} ≈ 1.16e-4` (ADR §3.4.2). Implement as a precomputed buffer.
- For position `m`, rotate each pair by angle `m · θ_i`.
- **Efficient implementation (§3.4.4)**: `q_even = q[..., 0::2]`, `q_odd = q[..., 1::2]`, then `rot_even = q_even·cos - q_odd·sin`, `rot_odd = q_even·sin + q_odd·cos`, interleave back via `tf.stack` + `tf.reshape` (the snippet in §3.4.4 is the binding reference).
- `cos`/`sin` tables precomputed once at model build time for `L = max(L_enc, L_dec)` and sliced per batch.

The ADR §3.4.5 application table is binding for downstream (it tells `MS2-MODEL-B-02` and `MS2-MODEL-B-03` exactly where to apply RoPE — cross-attention is ❌ across the board). This ticket implements only the layer; downstream wires it.

Required unit tests (ADR §3.11):

1. **Identity**: `apply_rope(x, cos=ones, sin=zeros) == x` within fp32 tolerance.
2. **Relative-position invariance**: for random `q, k` and shift `δ`, `<RoPE(q, m), RoPE(k, n)> == <RoPE(q, m+δ), RoPE(k, n+δ)>` within fp32 tolerance.
3. **Norm preservation**: `||RoPE(x, m)|| == ||x||` within fp32 tolerance.

These three tests are the same ones an evaluator can probe at the whiteboard (ADR §5 row "RoPE encodes relative position"); they double as defensive correctness checks.

### Acceptance Criteria

- `src/ms2/models/transformer/rope.py` implements `RoPE` as a `tf.keras.layers.Layer` subclass with precomputed `cos`/`sin` buffers.
- All three unit tests pass over a sweep of shapes and positions.
- `apply_rope(x, cos, sin)` accepts arbitrary leading batch dims (matches the snippet's `(B, n_heads, L, d_head)` shape contract).
- The frequency table matches the equation in ADR §3.4.2 within fp32 tolerance (verified at a few `i` values).
- `cos`/`sin` precomputation is build-time, not call-time.
- A short doc captures the RoPE contract — the application table from §3.4.5 is reproduced so consumers don't have to re-derive it.

### Produces Artifacts

- `src/ms2/models/transformer/rope.py`
- `tests/ms2/test_rope.py`
- "RoPE contract" doc with the §3.4.5 application table

### Blocking Class

Hard blocker (the entire transformer attention stack depends on this).

### Notes

- Dependencies: `MS2-INFRA-01` [DONE].
- Story points: 3.

---

## MS2-MODEL-B-02: Model B — Multi-Head Self-Attention, Cross-Attention, Transformer Encoder

### Description

Implement the attention machinery and the transformer encoder stack. ADR §3.11 explicitly forbids using `tf.keras.layers.MultiHeadAttention` because its position-encoding hooks cannot inject RoPE cleanly. Implement attention from `tf.matmul` and `tf.nn.softmax` directly.

Implementation contracts:

- **Multi-head self-attention** (used by encoder and by decoder masked self-attention):
  - Project `Q, K, V` from input, reshape to `(B, n_heads, L, d_head)`.
  - Apply RoPE to `Q` and `K` (NOT `V`, per ADR §3.4.5).
  - `scores = q · k^T / √d_head`.
  - Apply padding mask (set to `-inf` for pad keys); for the decoder variant, also apply causal mask (upper-triangular = `-inf`).
  - Softmax on `scores` in **float32** (cast up before softmax, cast back down before V mul, per ADR §3.11 last bullet).
  - Dropout `attn` with `dropout_attn = 0.1`.
  - `z = attn · v`, reshape, project with `Wo`.
- **Multi-head cross-attention**:
  - `Q` from decoder positions, `K, V` from encoder output.
  - **No RoPE applied to any of Q, K, V** (ADR §3.4.5 — the cross-attention table is the most-defended row of that table).
  - Padding mask applied on `K` positions (encoder padding).
  - Same float32 softmax and dropout pattern.
- **Encoder layer (pre-LayerNorm, §3.5)**:
  - Sub-layer 1 (self-attention with RoPE): `y=LN(x); q,k,v=Wq·y, Wk·y, Wv·y; q,k = RoPE; scores=...; ... ; x = x + Drop(z)`.
  - Sub-layer 2 (FFN): `y=LN(x); z = W2·GELU(W1·y+b1)+b2; x = x + Drop(z)`. `W1: 192 → 512`, `W2: 512 → 192`.
- **Encoder stack**: 3 layers; one final `LayerNorm` after the last layer (ADR §3.5 last paragraph).
- Hyperparameters per ADR §3.2: `d_model=192, d_ff=512, n_heads=4, d_head=48, n_enc_layers=3, dropout_attn=0.1, dropout_resid=0.1`.

This ticket also lands the **shared embedding** layer used by both encoder and decoder (ADR §3.3): `Embedding(V=4096, d_model=192)`, `RandomNormal(stddev=d_model**-0.5)`, scaled by `√d_model` after lookup, tied with the output projection (consumer in `MS2-MODEL-B-03`). Encoder and decoder share the same matrix.

### Acceptance Criteria

- `src/ms2/models/transformer/{attention, encoder, embedding}.py` implement self-attention, cross-attention (yes, here, even though only the decoder uses it — keeping all three layers in one file co-locates attention contracts), and the encoder stack.
- Padding mask correctness: `<pad>` keys receive zero attention weight (verified by a test with hand-crafted mask).
- Causal mask correctness (used later by decoder, but tested here): position `t` cannot attend to positions `> t`.
- Float32 softmax under mixed precision verified by test.
- RoPE applied to Q and K, not V (verified by inspection-test that asserts the call graph).
- Cross-attention applies _no_ RoPE on any of Q, K, V (verified by inspection-test).
- Encoder stack output shape `(B, L_enc, d_model=192)` for a `(B=2, L_enc=420)` mini-batch.
- Encoder param count within ±5 % of ADR §3.9 estimate (~1.30 M for 3 enc layers).
- Embedding scaled by `√d_model` after lookup; tied with output projection — verified by checking that `model.trainable_variables` does not double-count.

### Produces Artifacts

- `src/ms2/models/transformer/{attention, encoder, embedding}.py`
- `tests/ms2/test_transformer_attention.py`
- `tests/ms2/test_transformer_encoder.py`

### Blocking Class

Hard blocker.

### Notes

- Dependencies: `MS2-MODEL-B-01`.
- Story points: 4.

---

## MS2-MODEL-B-03: Model B — Transformer Decoder, K/V Cache, Full Assembly, Parameter Audit

### Description

Implement the transformer decoder, the inference-time K/V cache, the full Model B assembly, and the parameter-budget audit (ADR §3.9).

Decoder contracts (ADR §3.6):

- 3 stacked decoder layers, pre-LayerNorm.
- Sub-layer 1: masked self-attention with RoPE on Q and K; causal mask + decoder padding mask.
- Sub-layer 2: cross-attention (no RoPE); encoder padding mask on K positions.
- Sub-layer 3: FFN, same as encoder.
- After the 3rd decoder layer, one more `LayerNorm`, then output projection tied to the embedding (no extra `W_proj` because `d_model = d_tok = 192`, ADR §3.6 last paragraph).
- Output: `logits = decoder_output · E^T + b_out`.

K/V caching (ADR §3.8):

- During autoregressive decoding, the decoder's masked self-attention reuses cached K and V tensors from previous timesteps; only the new query position is computed.
- Cross-attention K and V are computed once per example (encoder runs once) and cached.
- Reduces inference complexity from O(L_dec²) to O(L_dec) per step.
- The implementation must support both training-mode (full sequence, no cache) and inference-mode (per-step, cache populated) without code duplication — a `cache: Optional[Mapping]` argument on the decoder layer is a clean pattern.

Note on the asymmetry called out in ADR §3.6 last block: Model A has `d_tok = 128 ≠ d_dec = 256` so it needs `W_proj`; Model B has `d_tok = d_model = 192` so it doesn't. This is intentional — do not "unify" by adding a no-op `W_proj` to Model B.

Parameter audit (ADR §3.9): produce a printed breakdown matching the §3.9 table; total within ±10 % of ~3.8 M.

### Acceptance Criteria

- `src/ms2/models/transformer/decoder.py` implements the 3-layer decoder with the three sub-layers described.
- Causal mask + decoder padding mask correctly composed (no leak from future positions, no leak from `<pad>`).
- Cross-attention reads encoder output, applies encoder padding mask on K, and is verified by a test on a hand-crafted batch.
- `src/ms2/models/transformer/kv_cache.py` exposes a small `KVCache` data class supporting "append step `t` to cache" and "produce full `K, K, V`-so-far". K/V cache reduces inference forward-pass FLOPs measured against a no-cache reference (verified by a microbenchmark in test).
- Decoder forward pass in training mode accepts `(B, L_dec, d_model)` and returns the same shape.
- Decoder forward pass in inference mode (per-step, cache supplied) returns `(B, 1, d_model)` for a single new query position.
- `src/ms2/models/transformer/model_b.py` exposes `ModelB(tf.keras.Model)` with `call(inputs, training)` taking `(encoder_input_ids, decoder_input_ids)` and returning `logits` plus auxiliary `(encoder_self_attn_weights, decoder_cross_attn_weights)` (used later for visualization in `MS2-COMPARE-01`).
- Forward-pass shape test on a `(B=2)` batch end-to-end.
- Parameter audit: printed breakdown matches ADR §3.9 categories; total within `[3.4M, 4.2M]`.
- Loss-decreases sanity check: 200 training steps on a tiny batch overfits training loss to near zero.

### Produces Artifacts

- `src/ms2/models/transformer/{decoder, kv_cache, model_b}.py`
- `tests/ms2/test_model_b_full.py`
- Parameter audit JSON + printed breakdown
- "Model B architecture card"

### Blocking Class

Hard blocker.

### Notes

- Dependencies: `MS2-MODEL-B-01`, `MS2-MODEL-B-02`.
- Story points: 4.

---

## MS2-INFER-01: Greedy and Beam-4 Decoding for Both Models

### Description

Implement the inference-time decoding for both Model A and Model B. ADR §2.13 (Model A) and §3.8 (Model B) specify:

- **Greedy decoding** as the headline number.
- **Beam search width 4**, length normalization with `α = 0.6`, as the secondary number.
- Maximum decode length `L_dec = 64`. Stop on `<eos>` or max length.
- For Model A: encoder output, encoder summary, FiLM `(γ, β)`, and decoder initial states are cached and reused across decoding steps (the encoder runs once per example).
- For Model B: encoder runs once per example; decoder uses the K/V cache from `MS2-MODEL-B-03` to reduce per-step cost.

Beam search:

- Maintain a beam of width 4.
- At each step, expand each beam by all V tokens, compute updated log-probabilities, keep top 4 by score.
- Length normalization: divide log-probability by `((5 + L) / 6)^α` with `α = 0.6` (the standard Wu et al. variant; the ADR doesn't pin the exact form, so use the standard form and document it).
- Beam terminates when all beams have emitted `<eos>` or hit `L_dec`.

For Model A's inference-time context windowing: when the test transcript is longer than `L_c`, generate per window with stride `L_c / 2`, then select the window's generation by **length-normalized log-probability** (ADR §1.3 last paragraph).

This ticket also benchmarks per-example greedy inference time and peak GPU memory for both models, recording numbers into `RunSummary`.

### Acceptance Criteria

- `src/ms2/inference/greedy.py` and `src/ms2/inference/beam.py` implement decoding for both Model A and Model B (model-agnostic via a small `DecodableModel` interface — `init_state(encoder_inputs) -> state; step(state, last_token) -> (logits, state)`).
- Greedy decoding hits `<eos>` correctly; verified on a tiny overfit model.
- Beam-4 decoding produces no worse log-probability than greedy on a tiny overfit model (sanity check on the ranking).
- Length normalization formula documented in code with the exact `((5+L)/6)^0.6` form.
- For Model A, context-windowing-with-stride implemented correctly for a transcript of length `> L_c`; the highest-confidence window is selected.
- For Model B, K/V cache is used (test asserts the cache is populated and the per-step forward pass receives only the new query position).
- Per-example greedy inference time and peak GPU memory recorded for both models on dev set.

### Produces Artifacts

- `src/ms2/inference/{greedy, beam, windowing}.py`
- `tests/ms2/test_inference.py`
- Per-model inference benchmark JSON

### Blocking Class

Hard blocker.

### Notes

- Dependencies: `MS2-MODEL-A-03`, `MS2-MODEL-B-03`.
- Story points: 3.

---

## MS2-TRAIN-02: Execute Training Runs (Both Models × 3 Seeds, Wall-Clock-Matched)

### Description

Run the actual training. Two models × three seeds (`{13, 42, 91}` per ADR §1.9) = six runs. Each run targets ~75 minutes wall-clock (ADR §2.12 / §3.7 / §4.1). The compute budget is **matched on wall-clock**, not parameter count or epoch count; ADR §4.1 explicitly says "epochs may differ; we report wall-clock as the primary axis".

For each run:

- Use `train_one_run(...)` from `MS2-TRAIN-01`.
- Save best-on-dev and last checkpoints to `experiments/ms2/<model>/<seed>/checkpoints/`.
- Save per-step training loss + per-epoch dev EM/F1/CED/BLEU-1 to `experiments/ms2/<model>/<seed>/curves/`.
- Save peak GPU memory (queried via `tf.config.experimental.get_memory_info('GPU:0')`) and total wall-clock to `RunSummary`.
- Confirm reproducibility: a follow-up dry-run with the same seed yields identical step-0 loss within fp32 tolerance.

If wall-clock budget is materially exceeded for any run, surface the deviation in the run summary; do _not_ silently extend.

### Acceptance Criteria

- Six `RunSummary` JSONs produced, one per (model, seed).
- Six checkpoint directories produced.
- Six training-curves files produced.
- Per-run wall-clock within ±10 % of the 75 min target, or deviation explicitly logged.
- Step-0 loss reproducible across two runs of the same seed.
- All runs use the schedule appropriate to their model (cosine-with-warmup for A, Noam for B — ADR §2.12 / §3.7).

### Produces Artifacts

- 6 × `experiments/ms2/<model>/<seed>/run_summary_v001.json`
- 6 × checkpoint directories
- 6 × training-curve files
- A consolidated `experiments/ms2/training_runs_index_v001.json`

### Blocking Class

Hard blocker (`MS2-EVAL-02`, `MS2-COMPARE-01`, and `MS2-REPORT-01` consume these).

### Notes

- Dependencies: `MS2-DATA-03`, `MS2-MODEL-A-03`, `MS2-MODEL-B-03`, `MS2-TRAIN-01`, `MS2-EVAL-01`. Compute-bound, not code-bound; story points reflect orchestration cost, not implementation.
- Story points: 3.

---

## MS2-EVAL-02: Evaluation Protocol — Noise Battery, Leave-2-Videos-Out, Long-Dependency, Difficulty Buckets, Conditioning Visualizations

### Description

This ticket implements the rest of the evaluation surface that ADR §4.4 requires for the diagnostic plots — everything beyond the basic metrics from `MS2-EVAL-01`. It is the meatiest single ticket in this milestone outside the model implementations themselves. The ADR §7 also mentions a deferred companion document `ms2_design_eval_protocol_v001.md`; producing that document is part of this ticket.

The four required components:

1. **Noise battery (§4.4 item 3)**: 5 noise types × multiple noise rates per type. Suggested noise types (to be finalized in the eval-protocol doc):
   - Character-level swap (transpose two adjacent characters at noise-rate r).
   - Character-level deletion at rate r.
   - Character-level insertion at rate r (sample a character from the corpus distribution).
   - Token-level masking at rate r (replace BPE token with `<unk>`).
   - Diacritic injection at rate r (insert random diacritic marks — a deliberately Arabic-specific noise).
     For each type, sweep `r ∈ {0.0, 0.05, 0.1, 0.2, 0.4}` and produce a Token-F1-vs-rate curve per model, both seeds aggregated.

2. **Leave-2-videos-out split (§4.3)**: in addition to the random split, hold out 2 of the 13 videos for test; train on the remaining 11. This is a **separate** training run protocol — but note that running 2 models × 3 seeds × 1 alternate split = 6 _additional_ training runs, doubling the compute spend. The ADR shows it in the headline table, so it's required. To keep cost bounded, the protocol may run **1 seed instead of 3** for the leave-2-videos-out variant if compute is tight; document the deviation.

3. **Long-dependency analysis (§4.4 item 2)**: bin dev examples by question→answer token distance in the context (distance = absolute index difference between question's first content token's nearest match in context and the answer span's start, or a similar surrogate to be defined in the protocol doc). Plot Token-F1 vs. distance, with confidence bands across seeds.

4. **Difficulty buckets (§4.4 item 4)**: read the `difficulty` field from QA CSVs (the CSV schema from `docs/fs/overall.md` §3 includes `difficulty`); bucket into easy/medium/hard; bar chart of Token-F1 per bucket per model.

5. **Conditioning visualizations (§4.4 item 5)**: cherry-pick 3–4 dev examples; for Model A, visualize the gated-merge weights (3 weights per context position) and FiLM `γ` values (256-dim per example); for Model B, visualize encoder self-attention (one head, one layer) and decoder cross-attention (one head, one layer). Heatmaps + a one-paragraph caption each.

### Acceptance Criteria

- `ms2_design_eval_protocol_v001.md` document is produced and lives at `docs/fs/ms2/ms2-eval-protocol.md` with: precise definitions of all noise types and rates, the long-dependency distance definition, the difficulty bucket boundaries, and the leave-2-videos-out video selection rationale.
- Noise battery results: 5 noise types × 5 rates × 2 models × ≥ 2 seeds = at minimum 100 evaluation cells; produced as a tidy DataFrame at `experiments/ms2/noise_battery_v001.csv`.
- Leave-2-videos-out training runs completed (1 seed minimum if compute-bound, 3 seeds preferred); `RunSummary`s produced.
- Long-dependency curves produced (one per model) with confidence bands.
- Difficulty bucket bar chart produced.
- 3–4 conditioning visualization figures produced — gated-merge / FiLM for Model A, encoder-self-attn / decoder-cross-attn for Model B.

### Produces Artifacts

- `docs/fs/ms2/ms2-eval-protocol.md`
- `experiments/ms2/noise_battery_v001.csv`
- `experiments/ms2/leave_2_videos_out/` (run summaries + checkpoints)
- `experiments/ms2/long_dependency_v001.json` and PNG figures
- `experiments/ms2/difficulty_buckets_v001.json` and PNG figure
- 3–4 conditioning visualization PNGs at `docs/fs/artifacts/ms2/`
- `src/ms2/analysis/{noise_battery, long_dependency, difficulty, conditioning_viz}.py`

### Blocking Class

Soft blocker for `MS2-COMPARE-01` (the comparison ticket can run on a subset; the full eval protocol output is required for the report).

### Notes

- Dependencies: `MS2-TRAIN-02`, `MS2-INFER-01`, `MS2-EVAL-01`.
- Story points: 5.

---

## MS2-ABLATE-01: Ablations for Both Models

### Description

Run all per-model ablations specified in ADR §2.15 (Model A) and §3.10 (Model B). Each ablation changes **exactly one thing** relative to the baseline (this is the core experimental design constraint).

Model A ablations (ADR §2.15):

1. **No-FiLM**: remove FiLM modulation; decoder receives only the initial state from `z`. Tests whether FiLM is doing real work.
2. **Mean-merge**: replace the gated softmax in §2.8 with a uniform mean over the three branches (`gates_i ≡ [1/3, 1/3, 1/3]`). Tests whether gating is doing real work.
3. **Plain Branch 3**: replace the char-CNN+BPE concat in Branch 3 (§2.6) with plain BPE embedding. Tests whether character features are doing real work.

Model B ablations (ADR §3.10):

1. **Sinusoidal-PE**: replace RoPE with classical additive sinusoidal positional encoding (the original "Attention Is All You Need" form), otherwise identical. Direct test of RoPE's contribution.
2. **No positional info**: drop both RoPE and any positional signal. Sanity-check ablation; expected to significantly degrade.
3. **Shared layers**: parameter sharing across the 3 encoder layers, separately across the 3 decoder layers. Tests whether depth or parameter count matters more. **Optional, run only if compute allows** (ADR §3.10 last line).

Each ablation must:

- Reuse the same training schedule, three seeds, same evaluation protocol as the baseline (ADR §2.15 last paragraph).
- Produce a `RunSummary` per (model, seed) under `experiments/ms2/ablations/<model>/<variant>/<seed>/`.
- Be invocable via `python -m src.cli.ms2 ablate --variant <name>` (`MS2-INFRA-03`).

The implementation strategy: each variant is a small wrapper around the baseline model that flips the relevant flag in the model-construction code. Use `AblationConfig` from `MS2-INFRA-02` to thread the variant through cleanly — do not duplicate model files.

### Acceptance Criteria

- All 5 mandatory ablations (A: no-FiLM, mean-merge, plain Branch 3; B: sinusoidal-PE, no-PE) trained for 3 seeds each = 15 runs. Optional 6th ablation (shared layers) may add 3 more.
- Each ablation differs from baseline by exactly one model-construction flag — verified by inspection (a small test that diffs the constructed model variables against baseline and asserts the expected delta).
- Ablation runs reuse `train_one_run(...)` and `RunSummary` schema unchanged.
- Sinusoidal-PE ablation: classical Vaswani sinusoidal encoding correctly implemented (added to embeddings before attention); tested.
- No-PE ablation: no positional signal whatsoever; verified by inspection.
- Mean-merge ablation: gates literally fixed to `[1/3, 1/3, 1/3]`, not learned to be uniform — this is the test of whether gating is doing real work, so the gates must be hard-coded constants.
- An `experiments/ms2/ablations_index_v001.json` consolidates all run summaries.

### Produces Artifacts

- 15 (or 18) × `experiments/ms2/ablations/.../run_summary_v001.json`
- 15 (or 18) × checkpoint directories
- `experiments/ms2/ablations_index_v001.json`
- `src/ms2/models/{rnn,transformer}/ablations.py` (variant flags / wrappers)

### Blocking Class

Non-blocker (parallel to `MS2-EVAL-02`; `MS2-REPORT-01` consumes these but can be drafted on partial results).

### Notes

- Dependencies: `MS2-MODEL-A-03`, `MS2-MODEL-B-03`, `MS2-TRAIN-01`, `MS2-TRAIN-02` (for compute-budget calibration). Compute-heavy.
- Story points: 4.

---

## MS2-COMPARE-01: Headline Comparison Table and Diagnostic Plots

### Description

Produce the report-ready comparison artifacts specified in ADR §4.3 and §4.4.

The headline table (ADR §4.3) has the following rows, all over both models with mean ± std across 3 seeds:

- Exact Match (random split)
- Token-F1 (random split)
- Char edit distance (random split)
- BLEU-1 (random split)
- Exact Match (leave-2-videos-out)
- Token-F1 (leave-2-videos-out)
- Parameter count (~2.7 M for A, ~3.8 M for B per ADR §2.14, §3.9)
- Training wall-clock (~75 min)
- Inference time / example (greedy)
- Peak GPU memory (training)

The diagnostic plots (ADR §4.4):

1. **Train/dev loss curves**, all 6 runs (2 models × 3 seeds), one figure with subplots per model.
2. **Token-F1 vs. question→answer token distance**, both models, with confidence bands across seeds. (Data computed by `MS2-EVAL-02`; this ticket produces the figure.)
3. **Token-F1 vs. noise rate**, one figure per noise type (5 noise types × 1 figure). Same: data from `MS2-EVAL-02`.
4. **Token-F1 by difficulty bucket**, bar chart.
5. **Attention/conditioning visualizations** on 3–4 cherry-picked examples — produced by `MS2-EVAL-02`, but this ticket assembles them into the report figure with consistent styling.

The output must be a small Python module `src/ms2/analysis/compare.py` that, given the run summaries and analysis JSONs, produces a headline-table CSV/Markdown and the figure set. The CLI entrypoint `python -m src.cli.ms2 compare` lives here.

### Acceptance Criteria

- Headline table CSV produced: `experiments/ms2/headline_comparison_v001.csv` with all rows from ADR §4.3 populated (or `—` if a cell genuinely has no data, e.g. leave-2-videos-out under compute-tight scenarios).
- Headline table Markdown produced: `docs/reports/ms2_headline_table.md` for direct embedding in the report.
- Figure 1 (loss curves, 6 runs) produced as a single figure with subplots.
- Figure 2 (long-dependency Token-F1) produced.
- Figures 3a–3e (Token-F1 vs. noise rate, one per noise type) produced.
- Figure 4 (difficulty bucket bar chart) produced.
- Figures 5a–5d (conditioning visualizations) produced and styled consistently.
- All figures use a consistent color palette per model (Model A and Model B always the same colors across figures).
- Each figure has a one-line caption suitable for direct quoting in the report.

### Produces Artifacts

- `experiments/ms2/headline_comparison_v001.csv`
- `docs/reports/ms2_headline_table.md`
- `docs/reports/figures/ms2_*.png` (≈ 11 figures)
- `src/ms2/analysis/compare.py`

### Blocking Class

Hard blocker for `MS2-REPORT-01`.

### Notes

- Dependencies: `MS2-TRAIN-02`, `MS2-EVAL-02`, `MS2-ABLATE-01` (optional for ablation deltas in the report).
- Story points: 3.

---

## MS2-REPORT-01: Milestone 2 Technical Report

### Description

Write the Milestone 2 technical report — 2 pages, Markdown, at `docs/reports/ms2_report.md`. The report follows the project-wide deliverable contract (`docs/fs/overall.md` §4: design choices, insights, output analysis, framework limitations) and the MS2-specific requirement that it address (per ADR-adjacent §6.4 of `docs/fs/ms2/milestone-2.md`):

- Why each model struggles or succeeds.
- Whether attention improves representation.
- Long-dependency handling comparison.
- Overfitting behavior.
- Computational trade-offs.

The "defensible architectural claims" cheat-sheet in ADR §5 is the source-of-truth phrasing for any architectural justification in the report — paraphrase from there for consistency with the whiteboard answers an evaluator will probe.

The report should embed the headline table (`MS2-COMPARE-01`) and select 4–5 figures (out of the ~11 produced). It should be defensible at the one-on-one discussion: every claim it makes is backed by an artifact in the repository.

### Acceptance Criteria

- `docs/reports/ms2_report.md` is approximately 2 pages.
- Headline comparison table embedded.
- 4–5 figures selected and embedded with captions.
- Required §6.4 dimensions (`milestone-2.md`) all addressed.
- Each architectural claim cross-references its artifact (run summary, figure, or model card) by relative path.
- A "limitations" section is included (compute budget, dataset size = 3.9 k examples, single-language scope, no pretrained models, etc.).
- A "what we'd do with more compute" section briefly notes the omitted ablations (e.g., shared layers if not run, Model A scheduled-sampling).

### Produces Artifacts

- `docs/reports/ms2_report.md`
- Curated figure selection (4–5 out of ~11)
- Engineering-insight notes (private dev-facing addendum, optional)

### Blocking Class

Hard blocker (final deliverable).

### Notes

- Dependencies: all MS2 implementation tickets, especially `MS2-COMPARE-01`. Cannot be drafted before `MS2-TRAIN-02` produces run summaries.
- Story points: 3.

---

## Recommended Execution Order

Strict dependency-respecting linearization. Items at the same level are parallelizable.

1. `MS2-INFRA-01` [DONE] Repository paths, mixed-precision policy, reproducibility helpers
2. `MS2-INFRA-02` Core MS2 schemas and run configuration
3. `MS2-INFRA-03` MS2 CLI interface
4. `MS2-DATA-01` Length distribution analysis and length-cap freeze
5. `MS2-DATA-02` BPE-4k tokenizer and character vocabulary
6. `MS2-DATA-03` tf.data input pipeline
7. **Parallel batch (a)**:
   - `MS2-TRAIN-01` Training utilities
   - `MS2-EVAL-01` Metrics implementation and Arabic post-normalization
8. **Parallel batch (b)** — Model A and Model B implementations can run in parallel:
   - Model A: `MS2-MODEL-A-01` → `MS2-MODEL-A-02` → `MS2-MODEL-A-03`
   - Model B: `MS2-MODEL-B-01` → `MS2-MODEL-B-02` → `MS2-MODEL-B-03`
9. `MS2-INFER-01` Greedy and beam-4 decoding
10. `MS2-TRAIN-02` Execute training runs (6 baseline runs)
11. **Parallel batch (c)**:
    - `MS2-EVAL-02` Eval protocol (noise battery, leave-2-videos-out, long-dep, difficulty, conditioning viz)
    - `MS2-ABLATE-01` Ablations
12. `MS2-COMPARE-01` Headline comparison table and diagnostic plots
13. `MS2-INFRA-04` End-to-end MS2 pipeline integration
14. `MS2-REPORT-01` Milestone 2 technical report

## Distribution to alice/bob/charly

This plan is the source-of-truth ticket inventory. Per-developer assignment lands in `docs/tickets/ms2/{alice,bob,charly}.md` after this plan is reviewed and frozen. Anticipated split (subject to revision):

- **One developer** owns Model A end-to-end (`MS2-MODEL-A-01..03`) plus its ablations subset.
- **One developer** owns Model B end-to-end (`MS2-MODEL-B-01..03`) plus its ablations subset.
- **One developer** owns the cross-cutting infrastructure, data pipeline, training utilities, evaluation, comparison, and report.

This split keeps the two architectural code-bases owned by single hands (so design intent is consistent) while letting the third developer move horizontally across the milestone — a structure that worked in MS1 (Charly's 6 cross-cutting tickets vs. Alice/Bob's stream-specific tickets).
