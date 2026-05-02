# MS2 Quick README

This is the fast map for where MS2 lives and how to run it.

## Main Commands

Install the normal dev environment:

```bash
uv sync --dev
```

Verify TensorFlow:

```bash
uv run python -c "import tensorflow as tf; print(tf.__version__)"
```

Run the full MS2 pipeline smoke:

```bash
uv run python -m src.cli.ms2 run-all --force-from analyze-lengths
```

Run tests:

```bash
uv run pytest tests/ms2
uv run pytest tests/cli
uv run ruff check .
uv run ruff format . --check
```

## Code Map

- `src/cli/ms2.py`: MS2 CLI entrypoint.
- `src/ms2/orchestration.py`: stage orchestration, resumable `run-all`, artifact writers.
- `src/common/paths.py`: canonical MS2 paths and filename helper.
- `src/ms2/schemas.py`: `RunConfig`, `BatchSpec`, `RunSummary`, `AblationConfig`.
- `src/ms2/data/`: tokenizer, length caps, data records, pipeline cache.
- `src/ms2/metrics/`: Arabic post-normalization and EM/F1/edit-distance/BLEU-1.
- `src/ms2/models/rnn/`: Model A, tri-encoder RNN + gated merge + FiLM decoder.
- `src/ms2/models/transformer/`: Model B, RoPE Transformer + KV cache.
- `src/ms2/inference/`: greedy, beam-4, sliding-window selection.
- `src/ms2/analysis/`: noise battery, long dependency, difficulty buckets, conditioning visualization hooks, comparison table.
- `src/ms2/training/`: optimizer, schedules, loss, training loop utilities.

## Artifact Map

- `data/processed/ms2/ms2_tokenizer_bpe_4k_v001.model`: SentencePiece BPE tokenizer.
- `data/processed/ms2/ms2_char_vocab_v001.json`: deterministic char vocabulary.
- `data/processed/ms2/ms2_tfrecords_train_v001.dir/`: cached pipeline output marker/files.
- `experiments/ms2/run_all_log_v001.txt`: first end-to-end run log.
- `experiments/ms2/model_a/<seed>/run_summary_v001.json`: Model A run summaries.
- `experiments/ms2/model_b/<seed>/run_summary_v001.json`: Model B run summaries.
- `experiments/ms2/training_runs_index_v001.json`: baseline run index.
- `experiments/ms2/noise_battery_v001.csv`: noise protocol table.
- `experiments/ms2/long_dependency_v001.json`: long-dependency protocol output.
- `experiments/ms2/difficulty_buckets_v001.json`: difficulty protocol output.
- `experiments/ms2/ablations_index_v001.json`: ablation run index.
- `docs/reports/ms2_headline_table.md`: comparison table.
- `docs/reports/ms2_report.md`: milestone report.
- `docs/reports/ms2_first_e2e_analysis.md`: first e2e run analysis.

## Important Limitation

The current `run-all` validates the complete pipeline wiring and artifact contracts. It does not perform six real 75-minute training jobs. Training/eval outputs are placeholder-safe and explicitly marked compute-bound until real checkpoints are produced.
