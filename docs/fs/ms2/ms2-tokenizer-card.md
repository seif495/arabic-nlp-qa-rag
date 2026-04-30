# MS2 Tokenizer Card

`MS2-DATA-01` freezes the tokenizer at `data/processed/ms2/ms2_tokenizer_bpe_4k_v001.model`.

Contract:

- Vocabulary size is exactly 4096.
- Special token IDs are fixed: `<pad>` = 0, `<unk>` = 1, `<bos>` = 2, `<eos>` = 3, `<sep>` = 4.
- Training corpus is transcript text only: `normalized_context` from the MS1 processed JSONL export.
- QA question/answer text is excluded from the tokenizer training corpus.
- Character vocabulary is written to `data/processed/ms2/ms2_char_vocab_v001.json` with deterministic codepoint ordering after `<pad>` and `<unk>`.
- The Python wrapper is `src.ms2.data.tokenizer` with `encode`, `decode`, and `encode_chars`.

The repository does not currently declare SentencePiece as a dependency, so the implementation is a deterministic stdlib-backed tokenizer proxy that preserves the special-ID, transcript-only, vocabulary-size, and wrapper contracts expected by downstream MS2 tickets.
