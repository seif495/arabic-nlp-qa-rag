# MS2 Artifact Map

MS2 code must use `resolve_ms2_paths(...)` and `make_ms2_output_filename(...)` from `src/common/paths.py` instead of inventing paths inline.

## Canonical Directories

| Name | Path | Purpose |
| --- | --- | --- |
| Cleaned input root | `data/external/ms2-cleaned-input` | Real MS2 input JSON files using the layout shown by `example.json.example`; `*.example` files are templates only and must not be consumed. |
| Processed data root | `data/processed/ms2` | Tokenizer, character vocabulary, TFRecord cache, and length-analysis outputs. |
| Experiment root | `experiments/ms2` | Per-model, per-seed run outputs. |
| Run output dir | `experiments/ms2/<model>/<seed>` | Training curves, checkpoints, evaluation JSON, and run summaries. |
| Report dir | `docs/fs/ms2` | MS2 contracts and artifact maps. |

## Named Files

| Artifact | Resolver Field | Default Path |
| --- | --- | --- |
| BPE tokenizer | `tokenizer_path` | `data/processed/ms2/ms2_tokenizer_bpe_4k_v001.model` |
| Character vocabulary | `char_vocab_path` | `data/processed/ms2/ms2_char_vocab_v001.json` |
| TFRecord shard directory | `tfrecord_shard_dir` | `data/processed/ms2/ms2_tfrecords_train_v001.dir` |

## Filename Contract

MS2 generated filenames use `ms2_<stage>_<name>_v###.<ext>`. The `stage` and `name` tokens must be lowercase snake_case alphanumeric strings with underscores only, and versions start at `v001`.

## Path Resolution Example

See `docs/fs/artifacts/ms2/ms2-path-resolution-example.txt` for the canonical manifest emitted by `resolve_ms2_paths().as_relative_manifest()`.

## Cleaned Input Selection

Use `list_ms2_cleaned_input_files(resolve_ms2_paths(...))` to discover MS2 input JSON files. The helper reads from `data/external/ms2-cleaned-input`, returns only real `.json` files, and deliberately excludes template files ending in `.example`, including `example.json.example` and `example.txt.example`.
