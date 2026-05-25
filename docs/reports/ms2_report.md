# Milestone 2 Report

MS2 implements two from-scratch Arabic QA generators following `docs/fs/ms2/ADR.md`: Model A, a no-attention tri-encoder recurrent model with gated fusion and FiLM conditioning, and Model B, a RoPE encoder-decoder Transformer with direct matmul attention.

## Headline Table

See `docs/reports/ms2_headline_table.md`. Empirical score cells are intentionally blank because full 6-seed training could not be run in this disk-constrained environment. No model scores, curves, or benchmark numbers are fabricated.

## Design Findings

Model A should struggle when answers require precise token-level retrieval over long contexts because its decoder receives only global encoder summary and FiLM signals. Its strengths are low parameter count, explicit branch specialization, and interpretable per-position branch gates.

Model B should handle long dependencies better because self-attention and cross-attention expose direct position-to-position paths. RoPE preserves relative-position information in self-attention while cross-attention intentionally receives no RoPE because decoder and encoder positions live in different frames.

## Compute And Limitations

Implementation paths are complete for metrics, Model A, Model B, inference, ablation flags, evaluation protocol hooks, comparison table generation, and CLI orchestration. Full baseline training, ablation training, leave-2-videos-out training, and diagnostic plots require TensorFlow plus enough disk/GPU resources. TensorFlow installation failed locally while extracting the wheel with `No space left on device`, so TensorFlow-dependent tests are present but skipped here.

## More Compute

With more compute, run 2 models x 3 seeds for 75 minutes each, then run mandatory ablations (`no_film`, `mean_merge`, `plain_branch3`, `sinusoidal_pe`, `no_pe`) and the leave-2-videos-out protocol. The final report should then replace placeholder-safe tables and figures with generated metrics, loss curves, noise curves, long-dependency plots, difficulty plots, and conditioning visualizations.
