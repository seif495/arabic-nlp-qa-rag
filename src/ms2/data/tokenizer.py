from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import sentencepiece as spm

from src.common.paths import resolve_ms2_paths
from src.ms2.data.length_caps import proxy_tokenize
from src.ms2.data.records import MS2DatasetRecord, load_ms1_processed_records


VOCAB_SIZE = 4096
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>", "<sep>")
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SEP_ID = 4
CHAR_PAD = "<pad>"
CHAR_UNK = "<unk>"
DEFAULT_MAX_CHARS = 16


class MS2Tokenizer:
    def __init__(
        self,
        id_to_piece: list[str],
        char_to_id: dict[str, int],
        processor: spm.SentencePieceProcessor,
    ) -> None:
        if len(id_to_piece) != VOCAB_SIZE:
            raise ValueError(f"tokenizer vocab must contain exactly {VOCAB_SIZE} pieces")
        for expected_id, token in enumerate(SPECIAL_TOKENS):
            if id_to_piece[expected_id] != token:
                raise ValueError(f"special token {token} must have id {expected_id}")
        if char_to_id.get(CHAR_PAD) != 0:
            raise ValueError("character <pad> id must be 0")
        self.id_to_piece = tuple(id_to_piece)
        self.piece_to_id = {piece: index for index, piece in enumerate(id_to_piece)}
        self.char_to_id = dict(char_to_id)
        self.processor = processor

    def encode(self, text: str) -> list[int]:
        return list(self.processor.encode(text, out_type=int))

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        encoded = self.processor.encode_as_immutable_proto(text)
        return [(piece.begin, piece.end) for piece in encoded.pieces]

    def decode(self, ids: list[int]) -> str:
        filtered_ids = [
            token_id
            for token_id in ids
            if token_id not in (PAD_ID, BOS_ID, EOS_ID, SEP_ID)
        ]
        return self.processor.decode(filtered_ids)

    def encode_chars(self, token_str: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[int]:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        encoded = [self.char_to_id.get(ch, self.char_to_id[CHAR_UNK]) for ch in token_str[:max_chars]]
        return encoded + [self.char_to_id[CHAR_PAD]] * (max_chars - len(encoded))


_DEFAULT_TOKENIZERS: dict[str, MS2Tokenizer] = {}


def train_tokenizer_assets(
    records: list[MS2DatasetRecord],
    repo_root: Path | None = None,
) -> MS2Tokenizer:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    transcript_texts = [record.normalized_context for record in records if record.normalized_context]
    pieces = _train_sentencepiece_pieces(transcript_texts, paths.tokenizer_path)
    char_vocab = build_char_vocab(transcript_texts)
    paths.char_vocab_path.write_text(json.dumps(char_vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    corpus_path = paths.data_processed_ms2 / "ms2_tokenizer_training_corpus_v001.json"
    corpus_path.write_text(json.dumps(transcript_texts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    processor = spm.SentencePieceProcessor(model_file=str(paths.tokenizer_path))
    return MS2Tokenizer(pieces, char_vocab, processor)


def load_tokenizer(repo_root: Path | None = None) -> MS2Tokenizer:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=False)
    char_vocab = json.loads(paths.char_vocab_path.read_text(encoding="utf-8"))
    processor = spm.SentencePieceProcessor(model_file=str(paths.tokenizer_path))
    pieces = _pad_pieces_to_vocab(
        [processor.id_to_piece(index) for index in range(processor.get_piece_size())]
    )
    return MS2Tokenizer(
        pieces,
        {str(k): int(v) for k, v in char_vocab.items()},
        processor,
    )


def ensure_default_tokenizer(repo_root: Path | None = None) -> MS2Tokenizer:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    cache_key = str(paths.repo_root)
    if cache_key in _DEFAULT_TOKENIZERS:
        return _DEFAULT_TOKENIZERS[cache_key]
    if paths.tokenizer_path.exists() and paths.char_vocab_path.exists():
        try:
            _DEFAULT_TOKENIZERS[cache_key] = load_tokenizer(repo_root=repo_root)
            return _DEFAULT_TOKENIZERS[cache_key]
        except Exception:
            pass
    records = load_ms1_processed_records(paths.repo_root / "data/processed/ms1/ms1_dataset_processed_v001.jsonl")
    _DEFAULT_TOKENIZERS[cache_key] = train_tokenizer_assets(
        records, repo_root=paths.repo_root
    )
    return _DEFAULT_TOKENIZERS[cache_key]


def encode(text: str) -> list[int]:
    return ensure_default_tokenizer().encode(text)


def decode(ids: list[int]) -> str:
    return ensure_default_tokenizer().decode(ids)


def encode_chars(token_str: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[int]:
    return ensure_default_tokenizer().encode_chars(token_str, max_chars=max_chars)


def build_char_vocab(texts: list[str]) -> dict[str, int]:
    chars = sorted({ch for text in texts for ch in text})
    vocab = {CHAR_PAD: 0, CHAR_UNK: 1}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def verify_special_token_ids(tokenizer: MS2Tokenizer) -> bool:
    return all(tokenizer.piece_to_id[token] == index for index, token in enumerate(SPECIAL_TOKENS))


def verify_char_coverage(texts: list[str], tokenizer: MS2Tokenizer) -> bool:
    return all(ch in tokenizer.char_to_id for text in texts for ch in text)


def _build_vocab_pieces(texts: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(proxy_tokenize(text))
        counts.update(ch for ch in text if not ch.isspace())
    ranked = [piece for piece, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    pieces = list(SPECIAL_TOKENS)
    for piece in ranked:
        if piece not in pieces:
            pieces.append(piece)
        if len(pieces) == VOCAB_SIZE:
            break
    while len(pieces) < VOCAB_SIZE:
        pieces.append(f"<unused_{len(pieces):04d}>")
    return pieces


def _train_sentencepiece_pieces(texts: list[str], model_path: Path) -> list[str]:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    training_texts = _sentencepiece_training_lines(texts)
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "spm_input.txt"
        input_path.write_text("\n".join(training_texts) + "\n", encoding="utf-8")
        prefix = Path(temp_dir) / "spm_model"
        spm.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=str(prefix),
            model_type="bpe",
            vocab_size=VOCAB_SIZE,
            character_coverage=1.0,
            num_threads=1,
            shuffle_input_sentence=False,
            pad_id=PAD_ID,
            unk_id=UNK_ID,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            bos_piece="<bos>",
            eos_piece="<eos>",
            user_defined_symbols=["<sep>"],
            hard_vocab_limit=False,
            max_sentence_length=20000,
        )
        trained_model = prefix.with_suffix(".model")
        model_path.write_bytes(trained_model.read_bytes())
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    return _pad_pieces_to_vocab(
        [processor.id_to_piece(index) for index in range(processor.get_piece_size())]
    )


def _pad_pieces_to_vocab(pieces: list[str]) -> list[str]:
    padded = list(pieces)
    while len(padded) < VOCAB_SIZE:
        padded.append(f"<unused_{len(padded):04d}>")
    return padded[:VOCAB_SIZE]


def _sentencepiece_training_lines(texts: list[str]) -> list[str]:
    lines: list[str] = []
    for text in texts:
        tokens = text.split()
        if not tokens:
            continue
        for start in range(0, len(tokens), 512):
            lines.append(" ".join(tokens[start : start + 512]))
    return lines or ["empty_corpus_placeholder"]
