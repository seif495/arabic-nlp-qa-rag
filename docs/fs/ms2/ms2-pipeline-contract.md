# MS2 Pipeline Contract

`MS2-DATA-01` exposes the data pipeline in `src.ms2.data.pipeline`.

Sequence formatting:

- Encoder input: `[<bos>] question_tokens [<sep>] context_tokens [<eos>]`, padded to `L_enc` with ID 0.
- Decoder input: `[<bos>] answer_tokens`, padded to `L_dec` with ID 0.
- Decoder target: `answer_tokens [<eos>]`, padded to `L_dec` with ID 0.
- Loss mask is true where decoder target ID is not 0.

Model schemas:

- Target model `a` includes `char_matrix` shaped `(L_enc, 16)`.
- Target model `b` omits `char_matrix`.

Context windows:

- Training windows center on the answer span with bounded jitter of `±20% * L_c`.
- Dev/test and inference windows are deterministic.
- Inference sliding windows use stride `L_c / 2` when a context exceeds `L_c`.

Caching:

- `prep-data --target-model {a,b}` writes split/model caches under canonical MS2 processed-data directories.
- Existing cache files are reused on subsequent calls.

Bucketing:

- Boundaries are `[128, 192, 256, 320, 420]`.
- Batch sizes scale inversely with boundary length to target approximately 16,384 tokens per batch.
