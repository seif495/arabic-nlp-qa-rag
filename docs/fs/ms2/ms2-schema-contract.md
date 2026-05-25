# MS2 Schema Contract

Source ticket: `MS2-INFRA-02`. Importable definitions live in `src/ms2/schemas.py`.

## RunConfig

Frozen-at-launch training configuration. JSON example: `docs/fs/artifacts/ms2/ms2-run-config.example.json`.

| Field | Type | ADR source | Meaning |
| --- | --- | --- | --- |
| `model_id` | `"A" \| "B"` | §1.3, §2, §3 | Model family bound to the run. |
| `seed` | `int` | §1.9 | Reproducibility seed. |
| `l_q` | `int` | §1.3 | Question length cap. |
| `l_c` | `int` | §1.3 | Context length cap. |
| `l_enc` | `int` | §1.3 | Encoder length cap. |
| `l_dec` | `int` | §1.3 | Decoder length cap. |
| `bucket_boundaries` | `list[int]` | §1.4 | Sequence-length bucket boundaries. |
| `target_tokens_per_batch` | `int` | §1.4 | Approximate token budget per batch. |
| `lr_schedule` | `"cosine_with_warmup" \| "noam"` | §2.12, §3.7 | Learning-rate schedule choice. |
| `wall_clock_budget_minutes` | `int` | §4.1 | Matched compute budget per training run. |
| `label_smoothing` | `float` | §1.7 | Cross-entropy label-smoothing value. |
| `gradient_clip_norm` | `float` | §2.12, §3.7 | Global norm clipping threshold. |

## BatchSpec

Declarative tensor-shape contract for a batch. JSON example: `docs/fs/artifacts/ms2/ms2-batch-spec.example.json`.

| Field | Type | ADR source | Meaning |
| --- | --- | --- | --- |
| `batch_size` | `int` | §1.4 | Number of examples in the batch. |
| `encoder_input_shape` | `list[int]` | §1.3 | Encoder input tensor shape. |
| `decoder_input_shape` | `list[int]` | §1.3 | Decoder input tensor shape. |
| `decoder_target_shape` | `list[int]` | §1.3 | Decoder target tensor shape. |
| `char_matrix_shape` | `list[int]` | §2.6 | Model A per-token character matrix shape. |
| `padding_id` | `int` | §1.1 | Padding token ID; fixed to `0`. |
| `l_char_max` | `int` | §2.6 | Maximum characters per BPE token; fixed to `16`. |

## RunSummary

Result-side metrics and resource summary for comparison-table loading. JSON example: `docs/fs/artifacts/ms2/ms2-run-summary.example.json`.

| Field | Type | ADR source | Meaning |
| --- | --- | --- | --- |
| `run_id` | `str` | §4.3 | Stable run identifier for downstream comparison. |
| `model_id` | `"A" \| "B"` | §4.3 | Model family summarized. |
| `seed` | `int` | §1.9 | Run seed. |
| `dev_em` | `float` | §1.8 | Development exact match. |
| `dev_token_f1` | `float` | §1.8 | Development token F1. |
| `dev_char_edit_distance` | `float` | §1.8 | Development normalized character edit distance. |
| `dev_bleu1` | `float` | §1.8 | Development BLEU-1. |
| `test_em` | `float` | §1.8 | Test exact match. |
| `test_token_f1` | `float` | §1.8 | Test token F1. |
| `test_char_edit_distance` | `float` | §1.8 | Test normalized character edit distance. |
| `test_bleu1` | `float` | §1.8 | Test BLEU-1. |
| `parameter_count` | `int` | §4.1, §4.3 | Trainable parameter count. |
| `peak_gpu_memory_mb` | `float` | §4.1, §4.3 | Peak GPU memory in MB. |
| `train_wall_clock_minutes` | `float` | §4.1, §4.3 | Training wall-clock duration. |
| `mean_inference_time_ms_per_example` | `float` | §4.1, §4.3 | Mean inference latency per example. |

## AblationConfig

Lightweight extension over `RunConfig` that records a single flipped ablation switch. JSON example: `docs/fs/artifacts/ms2/ms2-ablation-config.example.json`.

| Field | Type | ADR source | Meaning |
| --- | --- | --- | --- |
| `base_run_config` | `RunConfig` | §1.3, §1.4, §1.7, §1.9, §2.12, §3.7, §4.1 | Run configuration being modified. |
| `variant` | `str` | Ablation plan | One of `no_film`, `mean_merge`, `plain_branch3`, `sinusoidal_pe`, `no_pe`, `shared_layers`. |

## Serialization

Each schema exposes `to_dict()` and `from_dict(...)` for JSON-serializable round-trips. `RunConfig.frozen_dict()` returns a hashable tuple representation of every binding field so run identity changes when any field changes.
