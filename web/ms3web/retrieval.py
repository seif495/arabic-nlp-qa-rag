"""Lexical BM25 retriever over the exported MS3 transcript chunks.

The Streamlit build embedded chunks with ``paraphrase-multilingual-MiniLM``
inside ChromaDB. PyTorch + sentence-transformers exceed the Vercel function
size limit, so the web build ranks the same chunks with Okapi BM25 over
light-stemmed Arabic tokens instead. Model B (RoPE) can then re-rank the
candidates, mirroring the MS3 ``MS2Reranker`` flow.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ms3web.normalize import arabic_post_normalize

_TOKEN = re.compile(r"\w+", re.UNICODE)
_TATWEEL = "ـ"
# Longest first so "وال" wins over "و".
_PREFIXES = ("وبال", "وال", "بال", "كال", "فال", "لل", "ال")
_SUFFIXES = ("هما", "كما", "ات", "ون", "ين", "ها", "هم", "كم", "نا", "ة")
_STOPWORDS = frozenset(
    arabic_post_normalize(word)
    for word in """
    في من على الى إلى عن ان أن إن كان كانت هو هي هم دا دي ده اللي الذي التي
    و او أو ثم بس مش كده كدا يعني ايه إيه ما ماذا لماذا ليه هل لا لم لن مع عند
    كل بعد قبل حتى انا أنا انت احنا هذا هذه ذلك تلك هنا لكن يا ال
    the a an of to in on is are was were what who why how which and or
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Normalise (MS2 post-normalisation), drop stopwords, light-stem Arabic."""
    text = arabic_post_normalize(text).replace(_TATWEEL, "").lower()
    terms: list[str] = []
    for token in _TOKEN.findall(text):
        if token in _STOPWORDS or len(token) < 2:
            continue
        terms.append(_stem(token))
    return terms


def _stem(token: str) -> str:
    for prefix in _PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= 2:
            token = token[len(prefix) :]
            break
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    return token


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    episode: str
    text: str


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    rope_score: float | None = None


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        # Episode titles are indexed with the chunk so topic words match.
        self.doc_terms = [
            Counter(tokenize(f"{chunk.episode} {chunk.text}")) for chunk in chunks
        ]
        self.doc_len = [sum(terms.values()) for terms in self.doc_terms]
        self.avg_len = sum(self.doc_len) / max(len(self.doc_len), 1)
        df: Counter[str] = Counter()
        for terms in self.doc_terms:
            df.update(terms.keys())
        n = len(chunks)
        self.idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    @classmethod
    def from_json(cls, path: str | Path) -> BM25Retriever:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Chunk(**item) for item in payload["chunks"]])

    def search(self, query: str, k: int = 8) -> list[ScoredChunk]:
        query_terms = Counter(tokenize(query))
        if not query_terms:
            return []
        scored: list[ScoredChunk] = []
        for idx, terms in enumerate(self.doc_terms):
            score = 0.0
            norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[idx] / self.avg_len)
            for term in query_terms:
                tf = terms.get(term)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1.0) / (tf + norm)
            if score > 0.0:
                scored.append(ScoredChunk(self.chunks[idx], score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]
