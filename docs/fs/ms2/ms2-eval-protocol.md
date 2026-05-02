# MS2 Evaluation Protocol

This document implements the deferred ADR §7 protocol note for ADR §4.4 diagnostics.

Noise battery:

- Noise types: `char_swap`, `char_delete`, `char_insert`, `token_mask`, `diacritic_inject`.
- Rates: `{0.0, 0.05, 0.1, 0.2, 0.4}`.
- Output shape: tidy rows `(model, seed, noise_type, rate, token_f1, status)` in `experiments/ms2/noise_battery_v001.csv`.

Leave-2-videos-out:

- Hold out 2 videos as an alternate split. Preferred runs are 2 models x 3 seeds; if compute-bound, 1 seed is permitted with explicit report note.
- This implementation writes protocol hooks and placeholder-safe manifests; no trained leave-2-video checkpoints are fabricated.

Long dependency:

- Distance is the minimum absolute distance between the answer-start token index and any context token matching a non-trivial question token.
- Output: `experiments/ms2/long_dependency_v001.json` plus downstream figures when scored predictions exist.

Difficulty buckets:

- Buckets are `easy`, `medium`, `hard`, read from the QA `difficulty` field with Arabic/English aliases normalized.
- Output: `experiments/ms2/difficulty_buckets_v001.json` plus figure when scored predictions exist.

Conditioning visualizations:

- Model A: gated-merge weights and FiLM gamma.
- Model B: encoder self-attention and decoder cross-attention.
- This environment lacks trained checkpoints, so generated PNGs are placeholder-safe and labeled in the report as missing empirical visualizations.
