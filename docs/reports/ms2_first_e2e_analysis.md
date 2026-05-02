# MS2 First End-to-End Run Analysis

## Run Summary

Command executed:

```bash
uv run --extra ml python -m src.cli.ms2 run-all --force-from analyze-lengths
```

TensorFlow was installed through the optional `ml` extra after clearing the `uv` cache. The verified TensorFlow version was `2.21.0`.

The full MS2 pipeline smoke completed successfully. The canonical log is `experiments/ms2/run_all_log_v001.txt`.

## Stage Timings

From `experiments/ms2/run_all_log_v001.txt`:

| Stage | Count | Representative outputs | Wall-clock |
| --- | ---: | --- | ---: |
| `analyze-lengths` | 1 | `docs/fs/ms2/ms2-length-analysis-contract.md` | 81.995581s |
| `prep-data` | 2 | `data/processed/ms2/ms2_tfrecords_train_v001.dir/target_model_a`, `target_model_b` | 13.881638s, 112.565601s |
| `train` | 6 | `experiments/ms2/model_{a,b}/{13,42,91}/run_summary_v001.json` | placeholder-fast |
| `infer` | 6 | `dev_greedy_predictions.jsonl` per model/seed | placeholder-fast |
| `evaluate` | 6 | `dev_metrics.json` per model/seed | placeholder-fast |
| `evaluate-protocol` | 1 | `experiments/ms2/evaluation_protocol.json` | 0.014700s |
| `ablate` | 3 | Model A ablation run summaries for seed 13 | placeholder-fast |
| `compare` | 1 | `docs/reports/ms2_headline_table.md` | 0.003295s |

The slow stages were length analysis and data preparation because they scan/tokenize the MS1 processed dataset and retrain/cache SentencePiece/pipeline artifacts. The train/eval stages are intentionally fast because this run validated orchestration, schemas, paths, and artifact materialization, not the six 75-minute model training jobs.

## TensorFlow Model Validation

Command executed:

```bash
uv run --extra ml pytest tests/ms2/test_model_a_branches.py tests/ms2/test_model_a_middle.py tests/ms2/test_model_a_full.py tests/ms2/test_rope.py tests/ms2/test_transformer_attention.py tests/ms2/test_transformer_encoder.py tests/ms2/test_model_b_full.py
```

Result: `21 passed`.

The TensorFlow-backed validation covered:

- Model A branch shapes, structured pooling masks, char-CNN kernels `{2,3,4}`.
- Model A alignment with per-example question offsets, gated merge normalization, FiLM identity initialization, full forward pass, tied embedding projection, and tiny loss-decrease sanity check.
- RoPE identity, norm preservation, relative-position invariance, frequency table correctness, and arbitrary leading dims.
- Transformer self/cross attention masks, float32 softmax, RoPE applied only to self-attention Q/K, no RoPE in cross-attention.
- Transformer encoder output shape.
- Model B full forward pass, KV cache population, parameter audit, and tiny loss-decrease sanity check.

## Parameter Audits

Actual audits were generated with TensorFlow after model construction:

| Model | Audit artifact | Parameters | Required range | Status |
| --- | --- | ---: | ---: | --- |
| A | `docs/fs/artifacts/ms2/model_a_parameter_audit_v001.json` | 2,875,811 | 2.4M-3.0M | in range |
| B | `docs/fs/artifacts/ms2/model_b_parameter_audit_v001.json` | 3,407,344 | 3.4M-4.2M | in range |

Model B required a documented implementation adjustment: the direct Keras implementation with the ADR's FFN size `512` audited at 3,314,944 parameters, below the ticket's hard lower bound of 3.4M. The FFN hidden size was raised to `552`, bringing the model to 3,407,344 parameters while preserving depth, heads, RoPE placement, embedding tying, decoder/cache behavior, and `d_model=192`. This deviation is documented in `docs/fs/ms2/model-b-architecture-card.md`.

## Issues Found And Fixed During TF Run

- FiLM identity initialization was too noisy with `stddev=1e-3` for the test's `±0.01` bound over 100 vectors. The final FiLM projection now uses `stddev=1e-4`, preserving the ADR identity-init intent more strongly.
- Model B cache mutation was hidden by Keras symbolic tracing. Cache updates now happen only during eager execution and are copied back to `KVCache`, so step decoding visibly populates self-attention K/V.
- A symbolic causal-mask branch used a tensor as a Python boolean. The mask path was simplified to avoid graph-tracing warnings/errors.

## Artifact Completeness

Created or refreshed by the e2e run:

- `experiments/ms2/run_all_log_v001.txt`
- `experiments/ms2/ms2_pipeline_artifact_manifest_v001.json`
- `experiments/ms2/ms2_pipeline_integration_checklist_v001.md`
- `experiments/ms2/ms2_pipeline_known_limitations_v001.md`
- `experiments/ms2/training_runs_index_v001.json`
- `experiments/ms2/model_a/{13,42,91}/run_summary_v001.json`
- `experiments/ms2/model_b/{13,42,91}/run_summary_v001.json`
- `experiments/ms2/noise_battery_v001.csv`
- `experiments/ms2/long_dependency_v001.json`
- `experiments/ms2/difficulty_buckets_v001.json`
- `experiments/ms2/evaluation_protocol.json`
- `experiments/ms2/ablations_index_v001.json`
- `experiments/ms2/ablations/model_a/{no_film,mean_merge,plain_branch3}/13/run_summary_v001.json`
- `docs/reports/ms2_headline_table.md`

## What This Run Proves

- The CLI entrypoint works end to end through every MS2 stage.
- Canonical MS2 paths are used by orchestration outputs.
- TensorFlow/Keras model code imports, builds, runs forward passes, and satisfies the architecture shape/parameter/cache tests.
- Metrics and inference utilities pass concrete non-TensorFlow tests.
- Pipeline artifacts are materialized in the expected locations.
- Missing empirical results are represented honestly by placeholder-safe JSON/CSV files instead of fake metrics.

## What This Run Does Not Prove

- It does not prove final model quality, because no 75-minute training runs were executed.
- It does not produce real dev/test EM, F1, edit distance, BLEU-1, inference latency, or GPU memory numbers.
- It does not produce real noise-battery curves, long-dependency curves, difficulty bucket performance, or attention/conditioning visualizations.
- It does not execute the full mandatory ablation training matrix; only orchestration artifacts for the default `run-all` ablation subset are materialized.

## Next Real Experiment Step

Run one real `train_one_run` integration with a TensorFlow dataset and `ModelA` or `ModelB` for seed 13, then replace the placeholder `run_summary_v001.json`, curves, checkpoints, predictions, and metrics for that run. After that single-run path is verified, scale to the 6 baseline runs and then the ablation/eval protocol matrices.
