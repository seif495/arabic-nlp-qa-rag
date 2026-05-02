# MS2 Metric Contract

All MS2 evaluation metrics first apply `arabic_post_normalize` from `src/ms2/metrics/normalize.py` to both prediction and reference.

Normalization rules, per ADR §1.8:

- `أ`, `إ`, `آ` map to `ا`.
- `ى` maps to `ي`.
- Arabic Tashkeel and Quranic annotation marks are stripped.
- Runs of whitespace are collapsed so token metrics are stable.

Metric definitions:

- Exact Match: `1.0` iff normalized strings are equal, else `0.0`.
- Token-F1: SQuAD-style bag-of-words F1 over normalized whitespace tokens.
- Normalized character edit distance: Levenshtein distance divided by `max(len(pred), len(ref))`; empty/empty is `0.0`.
- BLEU-1: clipped unigram precision with standard brevity penalty; empty/empty is `1.0` and one-sided empty is `0.0`.

`aggregate_metrics(predictions, references)` returns `em`, `token_f1`, `char_edit_distance`, and `bleu1` keys for compatibility with `RunSummary` metric fields.
