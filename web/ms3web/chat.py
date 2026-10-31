"""Stateless RAG chat turn for the Vercel API.

Port of ``src/ms3/generation/chatbot.py``: the same memory strategies,
prompt strategies, out-of-domain handling and generation modes
(``llm_only`` / ``ms2_only`` / ``hybrid``). Serverless functions keep no
state between requests, so the browser sends the conversation history with
every turn.

The MS2 models (see ``ms3web/ms2.py``; the default is the ``select-ms2-web``
winner, the user may pick another) are used in two places:

- re-ranking: BM25 candidates are re-ordered by a 50/50 blend of their
  (min-max normalised) BM25 score and the cosine similarity of the MS2
  model's question vs. chunk encodings (cf. ``MS2Reranker``);
- generation: the MS2 model decodes an answer over sliding windows of the
  top chunks and keeps the window with the best length-normalised log-prob
  (``src/ms2/inference/windowing.py``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ms3web.llm import LLMClient, LLMUnavailableError
from ms3web.ms2 import MS2Runtime, load_runtimes
from ms3web.normalize import arabic_post_normalize
from ms3web.prompts import (
    HUMAN_TEMPLATE,
    STANDALONE_QUERY_TEMPLATE,
    SUMMARY_TEMPLATE,
    TEMPLATES,
    fill,
)
from ms3web.retrieval import BM25Retriever, ScoredChunk

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
CHUNKS_PATH = ASSETS_DIR / "ms3_chunks.json"

MEMORY_STRATEGIES = ("sliding_window", "full_history", "strict_truncation", "summarized_history")
GENERATION_MODES = ("llm_only", "ms2_only", "hybrid")
OUT_OF_DOMAIN_MESSAGE = (
    "عذراً، هذا السؤال خارج نطاق النصوص المتوفرة لدي.\n"
    "Sorry, this question is out of bounds for the available transcripts."
)
NO_LLM_MESSAGE = (
    "لم يتم إعداد مفتاح API لنموذج اللغة.\nNo LLM API key is configured on the "
    "server (set GROQ_API_KEY or GOOGLE_API_KEY). Switch to 'MS2 Only' to use "
    "the MS2 model alone."
)
LLM_ERROR_MESSAGE = (
    "حدث خطأ غير متوقع في النظام. الرجاء المحاولة مرة أخرى لاحقاً.\n"
    "An unexpected system error occurred. Please try again later."
)
RERANK_CANDIDATES = 30
MS2_RERANK_WEIGHT = 0.5
MS2_GENERATION_CHUNKS = 3


@dataclass
class ChatSettings:
    prompt_type: str = "arabic_guided"
    memory_strategy: str = "sliding_window"
    generation_mode: str = "llm_only"
    use_reranking: bool = False
    top_k: int = 6
    ms2_model: str | None = None  # None -> the selected (best) MS2 model


@dataclass
class ChatResult:
    answer: str
    llm_answer: str | None
    ms2_answer: str | None
    standalone_query: str
    sources: list[dict]
    logs: dict = field(default_factory=dict)


def format_history(
    history: list[tuple[str, str]], strategy: str, llm: LLMClient | None = None
) -> str:
    """Same four context-window strategies as ``RagChatbot.format_history``."""
    if not history:
        return ""
    if strategy == "full_history":
        turns = history
    elif strategy == "strict_truncation":
        raw = ""
        for human, ai in reversed(history):
            turn = f"User: {human}\nAssistant: {ai}\n"
            if len(raw) + len(turn) > 1000:
                break
            raw = turn + raw
        return raw
    elif strategy == "summarized_history" and llm is not None and llm.available:
        full = "".join(f"User: {human}\nAssistant: {ai}\n" for human, ai in history)
        try:
            summary = llm.complete("", fill(SUMMARY_TEMPLATE, history=full)).text
            return f"Summary of previous conversation:\n{summary}"
        except LLMUnavailableError:
            turns = history[-3:]
    else:  # sliding_window (and summary fallback without an LLM)
        turns = history[-3:]
    return "\n".join(
        line for human, ai in turns for line in (f"User: {human}", f"Assistant: {ai}")
    )


class RagEngine:
    """Loads the retriever and the MS2 models once per warm serverless instance."""

    def __init__(
        self,
        retriever: BM25Retriever,
        ms2_models: dict[str, MS2Runtime],
        default_ms2: str | None,
        llm: LLMClient,
    ) -> None:
        self.retriever = retriever
        self.ms2_models = ms2_models
        self.default_ms2 = default_ms2 if default_ms2 in ms2_models else None
        self.llm = llm

    @classmethod
    def from_assets(cls, llm: LLMClient | None = None) -> RagEngine:
        models, default = load_runtimes(ASSETS_DIR)
        return cls(BM25Retriever.from_json(CHUNKS_PATH), models, default, llm or LLMClient.from_env())

    @property
    def ms2_available(self) -> bool:
        return self.default_ms2 is not None

    def pick_ms2(self, name: str | None) -> MS2Runtime | None:
        if name in self.ms2_models:
            return self.ms2_models[name]
        return self.ms2_models.get(self.default_ms2) if self.default_ms2 else None

    # -- MS2 model ----------------------------------------------------------

    @staticmethod
    def ms2_rerank(
        ms2: MS2Runtime, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        q_vec = ms2.question_vector(query)
        c_vecs = ms2.context_vectors([item.chunk.text for item in candidates])
        rescored = [
            ScoredChunk(item.chunk, item.score, ms2_score=float(np.dot(q_vec, c_vec)))
            for item, c_vec in zip(candidates, c_vecs, strict=True)
        ]
        # Blend min-max-normalised BM25 and MS2 cosine so the from-scratch
        # model re-orders the lexical candidates instead of overriding them.
        bm25 = _min_max([item.score for item in rescored])
        ms2 = _min_max([item.ms2_score or 0.0 for item in rescored])
        order = sorted(
            range(len(rescored)),
            key=lambda i: (1 - MS2_RERANK_WEIGHT) * bm25[i] + MS2_RERANK_WEIGHT * ms2[i],
            reverse=True,
        )
        return [rescored[i] for i in order[:top_k]]

    @staticmethod
    def ms2_generate(ms2: MS2Runtime, question: str, chunks: list[ScoredChunk]) -> tuple[str, dict]:
        question = arabic_post_normalize(question)
        tokenizer = ms2.tokenizer
        windows, owners = [], []
        for item in chunks[:MS2_GENERATION_CHUNKS]:
            for window in tokenizer.context_windows(item.chunk.text):
                windows.append(window)
                owners.append(item.chunk)
        best = None
        for result, chunk in zip(ms2.generate(question, windows), owners, strict=True):
            if best is None or result.mean_log_prob > best[0].mean_log_prob:
                best = (result, chunk)
        if best is None or not best[0].token_ids:
            return "", {}
        result, chunk = best
        return tokenizer.decode(result.token_ids).strip(), {
            "episode": chunk.episode,
            "chunk_id": chunk.chunk_id,
            "mean_log_prob": round(result.mean_log_prob, 4),
        }

    # -- turn -------------------------------------------------------------

    def answer(
        self, question: str, history: list[tuple[str, str]], settings: ChatSettings
    ) -> ChatResult:
        timings: dict[str, float] = {}
        mode = settings.generation_mode if settings.generation_mode in GENERATION_MODES else "llm_only"
        ms2 = self.pick_ms2(settings.ms2_model)
        if mode != "llm_only" and ms2 is None:
            mode = "llm_only"
        use_reranking = settings.use_reranking and ms2 is not None
        top_k = max(1, min(int(settings.top_k), 15))
        llm_needed = mode in ("llm_only", "hybrid")
        llm_errors: list[str] = []
        provider = None

        started = time.perf_counter()
        history_text = format_history(history, settings.memory_strategy, self.llm)
        timings["history_ms"] = _ms(started)

        # 1. Standalone (contextualised, Arabic) retrieval query.
        started = time.perf_counter()
        standalone = question
        if self.llm.available:
            try:
                reply = self.llm.complete(
                    "",
                    fill(
                        STANDALONE_QUERY_TEMPLATE,
                        history=history_text if history else "No history. This is the first question.",
                        question=question,
                    ),
                )
                standalone = reply.text.strip() or question
                provider = f"{reply.provider}:{reply.model}"
                llm_errors.extend(reply.fallback_errors)
            except LLMUnavailableError as exc:
                llm_errors.append(str(exc))
        timings["rewrite_ms"] = _ms(started)

        # 2. Retrieve (+ optional MS2 re-ranking).
        started = time.perf_counter()
        retrieval_query = arabic_post_normalize(standalone)
        if use_reranking:
            candidates = self.retriever.search(f"{retrieval_query} {question}", k=RERANK_CANDIDATES)
            timings["retrieval_ms"] = _ms(started)
            started = time.perf_counter()
            docs = self.ms2_rerank(ms2, retrieval_query, candidates, top_k)
            timings["rerank_ms"] = _ms(started)
        else:
            docs = self.retriever.search(f"{retrieval_query} {question}", k=top_k)
            timings["retrieval_ms"] = _ms(started)
        context_text = "\n\n".join(item.chunk.text for item in docs)

        # 3. Generate.
        llm_answer = ms2_answer = None
        ms2_meta: dict = {}
        if llm_needed:
            started = time.perf_counter()
            if not self.llm.available:
                llm_answer = NO_LLM_MESSAGE
            else:
                system = TEMPLATES.get(settings.prompt_type, TEMPLATES["english_guided"])
                try:
                    reply = self.llm.complete(
                        fill(system, context=context_text),
                        fill(HUMAN_TEMPLATE, history=history_text, question=question),
                    )
                    provider = f"{reply.provider}:{reply.model}"
                    llm_errors.extend(reply.fallback_errors)
                    llm_answer = reply.text
                    if "OUT_OF_DOMAIN" in llm_answer.upper():
                        llm_answer = OUT_OF_DOMAIN_MESSAGE
                except LLMUnavailableError as exc:
                    llm_errors.append(str(exc))
                    llm_answer = LLM_ERROR_MESSAGE
            timings["llm_ms"] = _ms(started)
        if mode in ("ms2_only", "hybrid"):
            started = time.perf_counter()
            if docs:
                ms2_answer, ms2_meta = self.ms2_generate(ms2, standalone, docs)
            ms2_answer = ms2_answer or "[The MS2 model produced an empty answer]"
            timings["ms2_ms"] = _ms(started)

        answer = ms2_answer if mode == "ms2_only" else llm_answer
        return ChatResult(
            answer=answer or "",
            llm_answer=llm_answer,
            ms2_answer=ms2_answer,
            standalone_query=standalone,
            sources=[
                {
                    "episode": item.chunk.episode,
                    "chunk_id": item.chunk.chunk_id,
                    "bm25": round(item.score, 3),
                    "ms2_score": None if item.ms2_score is None else round(item.ms2_score, 4),
                    "preview": item.chunk.text[:280],
                }
                for item in docs
            ],
            logs={
                "history_text": history_text,
                "memory_strategy": settings.memory_strategy,
                "prompt_type": settings.prompt_type,
                "generation_mode": mode,
                "reranking": use_reranking,
                "retrieval_query": retrieval_query,
                "llm_provider": provider,
                "llm_errors": llm_errors,
                "ms2_model": ms2.info["model"] if ms2 is not None and (use_reranking or mode != "llm_only") else None,
                "ms2_window": ms2_meta,
                "history_turns": len(history),
                "timings_ms": timings,
            },
        )


def _min_max(values: list[float]) -> list[float]:
    low, high = min(values, default=0.0), max(values, default=0.0)
    if high - low < 1e-12:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 1)
