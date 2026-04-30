from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

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
    def __init__(self, id_to_piece: list[str], char_to_id: dict[str, int]) -> None:
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

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in proxy_tokenize(text):
            ids.append(self.piece_to_id.get(piece, UNK_ID))
        return ids

    def decode(self, ids: list[int]) -> str:
        pieces = []
        for token_id in ids:
            if token_id in (PAD_ID, BOS_ID, EOS_ID, SEP_ID):
                continue
            if 0 <= token_id < len(self.id_to_piece):
                piece = self.id_to_piece[token_id]
                if not piece.startswith("<unused_") and piece != "<unk>":
                    pieces.append(piece)
        return " ".join(pieces)

    def encode_chars(self, token_str: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[int]:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        encoded = [self.char_to_id.get(ch, self.char_to_id[CHAR_UNK]) for ch in token_str[:max_chars]]
        return encoded + [self.char_to_id[CHAR_PAD]] * (max_chars - len(encoded))


_DEFAULT_TOKENIZER: MS2Tokenizer | None = None


def train_tokenizer_assets(
    records: list[MS2DatasetRecord],
    repo_root: Path | None = None,
) -> MS2Tokenizer:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    transcript_texts = [record.normalized_context for record in records if record.normalized_context]
    pieces = _build_vocab_pieces(transcript_texts)
    char_vocab = build_char_vocab(transcript_texts)
    model_payload = {
        "type": "deterministic_transcript_bpe_proxy",
        "vocab_size": VOCAB_SIZE,
        "special_tokens": {token: index for index, token in enumerate(SPECIAL_TOKENS)},
        "id_to_piece": pieces,
        "training_corpus": "transcript_text_only",
    }
    paths.tokenizer_path.write_text(json.dumps(model_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths.char_vocab_path.write_text(json.dumps(char_vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    corpus_path = paths.data_processed_ms2 / "ms2_tokenizer_training_corpus_v001.json"
    corpus_path.write_text(json.dumps(transcript_texts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return MS2Tokenizer(pieces, char_vocab)


def load_tokenizer(repo_root: Path | None = None) -> MS2Tokenizer:
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=False)
    model_payload = json.loads(paths.tokenizer_path.read_text(encoding="utf-8"))
    char_vocab = json.loads(paths.char_vocab_path.read_text(encoding="utf-8"))
    return MS2Tokenizer(list(model_payload["id_to_piece"]), {str(k): int(v) for k, v in char_vocab.items()})


def ensure_default_tokenizer(repo_root: Path | None = None) -> MS2Tokenizer:
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is not None:
        return _DEFAULT_TOKENIZER
    paths = resolve_ms2_paths(repo_root=repo_root, create_dirs=True)
    if paths.tokenizer_path.exists() and paths.char_vocab_path.exists():
        _DEFAULT_TOKENIZER = load_tokenizer(repo_root=repo_root)
        return _DEFAULT_TOKENIZER
    records = load_ms1_processed_records(paths.repo_root / "data/processed/ms1/ms1_dataset_processed_v001.jsonl")
    _DEFAULT_TOKENIZER = train_tokenizer_assets(records, repo_root=paths.repo_root)
    return _DEFAULT_TOKENIZER


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
