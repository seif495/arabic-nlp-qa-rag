"""Stateless RAG chat turn for the Vercel API.

Port of ``src/ms3/generation/chatbot.py``: the same memory strategies,
prompt strategies, out-of-domain handling and generation modes
(``llm_only`` / ``ms2_only`` / ``hybrid``). Serverless functions keep no
state between requests, so the browser sends the conversation history with
every turn.

MS2 Model B (the RoPE Transformer) is used in two places:

- re-ranking: BM25 candidates are re-ordered by a 50/50 blend of their
  (min-max normalised) BM25 score and the cosine similarity of Model B's
  RoPE-encoder states for the question vs. each chunk (cf. ``MS2Reranker``);
- generation: Model B decodes an answer over sliding windows of the top
  chunks and keeps the window with the best length-normalised log-prob
  (``src/ms2/inference/windowing.py``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ms3web.llm import LLMClient, LLMUnavailableError
from ms3web.model_b import ModelBNumpy
from ms3web.normalize import arabic_post_normalize
from ms3web.prompts import (
    HUMAN_TEMPLATE,
    STANDALONE_QUERY_TEMPLATE,
    SUMMARY_TEMPLATE,
    TEMPLATES,
    fill,
)
from ms3web.retrieval import BM25Retriever, ScoredChunk
from ms3web.tokenizer import MS2Tokenizer

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
CHUNKS_PATH = ASSETS_DIR / "ms3_chunks.json"
MODEL_B_PATH = ASSETS_DIR / "ms2_model_b_rope.npz"
TOKENIZER_PATH = ASSETS_DIR / "ms2_tokenizer.model"

MEMORY_STRATEGIES = ("sliding_window", "full_history", "strict_truncation", "summarized_history")
GENERATION_MODES = ("llm_only", "ms2_only", "hybrid")
OUT_OF_DOMAIN_MESSAGE = (
    "عذراً، هذا السؤال خارج نطاق النصوص المتوفرة لدي.\n"
    "Sorry, this question is out of bounds for the available transcripts."
)
NO_LLM_MESSAGE = (
    "لم يتم إعداد مفتاح API لنموذج اللغة.\nNo LLM API key is configured on the "
    "server (set GROQ_API_KEY or GOOGLE_API_KEY). Switch to 'MS2 Only' to use "
    "the RoPE Transformer alone."
)
LLM_ERROR_MESSAGE = (
    "حدث خطأ غير متوقع في النظام. الرجاء المحاولة مرة أخرى لاحقاً.\n"
    "An unexpected system error occurred. Please try again later."
)
RERANK_CANDIDATES = 30
ROPE_RERANK_WEIGHT = 0.5
MS2_GENERATION_CHUNKS = 3


@dataclass
class ChatSettings:
    prompt_type: str = "arabic_guided"
    memory_strategy: str = "sliding_window"
    generation_mode: str = "llm_only"
    use_reranking: bool = False
    top_k: int = 6


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
    """Loads the retriever and Model B once per warm serverless instance."""

    def __init__(
        self,
        retriever: BM25Retriever,
        model_b: ModelBNumpy | None,
        tokenizer: MS2Tokenizer | None,
        llm: LLMClient,
    ) -> None:
        self.retriever = retriever
        self.model_b = model_b
        self.tokenizer = tokenizer
        self.llm = llm

    @classmethod
    def from_assets(cls, llm: LLMClient | None = None) -> RagEngine:
        model_b = tokenizer = None
        if MODEL_B_PATH.exists() and TOKENIZER_PATH.exists():
            model_b = ModelBNumpy.load(MODEL_B_PATH)
            tokenizer = MS2Tokenizer(TOKENIZER_PATH)
        return cls(
            BM25Retriever.from_json(CHUNKS_PATH),
            model_b,
            tokenizer,
            llm or LLMClient.from_env(),
        )

    @property
    def ms2_available(self) -> bool:
        return self.model_b is not None and self.tokenizer is not None

    # -- MS2 Model B (RoPE) -------------------------------------------------

    def rope_rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        assert self.model_b is not None and self.tokenizer is not None
        from ms3web.model_b import PAD_ID
        from ms3web.tokenizer import L_C, L_Q

        def encode(ids: list[int], cap: int) -> list[int]:
            ids = ids[:cap]
            return ids + [PAD_ID] * (cap - len(ids))

        q_vec = self.model_b.pooled_encoding(encode(self.tokenizer.encode(query), L_Q))
        rescored = []
        for item in candidates:
            c_vec = self.model_b.pooled_encoding(
                encode(self.tokenizer.encode(item.chunk.text), L_C)
            )
            rescored.append(
                ScoredChunk(item.chunk, item.score, rope_score=float(np.dot(q_vec, c_vec)))
            )
        # Blend min-max-normalised BM25 and RoPE cosine so the from-scratch
        # Model B re-orders the lexical candidates instead of overriding them.
        bm25 = _min_max([item.score for item in rescored])
        rope = _min_max([item.rope_score or 0.0 for item in rescored])
        order = sorted(
            range(len(rescored)),
            key=lambda i: (1 - ROPE_RERANK_WEIGHT) * bm25[i] + ROPE_RERANK_WEIGHT * rope[i],
            reverse=True,
        )
        return [rescored[i] for i in order[:top_k]]

    def ms2_generate(self, question: str, chunks: list[ScoredChunk]) -> tuple[str, dict]:
        assert self.model_b is not None and self.tokenizer is not None
        question = arabic_post_normalize(question)
        best = None
        for item in chunks[:MS2_GENERATION_CHUNKS]:
            for window in self.tokenizer.context_windows(item.chunk.text):
                result = self.model_b.greedy_decode(
                    self.tokenizer.encoder_input(question, window)
                )
                if best is None or result.mean_log_prob > best[0].mean_log_prob:
                    best = (result, item.chunk)
        if best is None or not best[0].token_ids:
            return "", {}
        result, chunk = best
        return self.tokenizer.decode(result.token_ids).strip(), {
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
        if mode != "llm_only" and not self.ms2_available:
            mode = "llm_only"
        use_reranking = settings.use_reranking and self.ms2_available
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
            except LLMUnavailableError as exc:
                llm_errors.append(str(exc))
        timings["rewrite_ms"] = _ms(started)

        # 2. Retrieve (+ optional Model B RoPE re-ranking).
        started = time.perf_counter()
        retrieval_query = arabic_post_normalize(standalone)
        if use_reranking:
            candidates = self.retriever.search(f"{retrieval_query} {question}", k=RERANK_CANDIDATES)
            timings["retrieval_ms"] = _ms(started)
            started = time.perf_counter()
            docs = self.rope_rerank(retrieval_query, candidates, top_k)
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
                ms2_answer, ms2_meta = self.ms2_generate(standalone, docs)
            ms2_answer = ms2_answer or "[Model B produced an empty answer]"
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
                    "rope_score": None if item.rope_score is None else round(item.rope_score, 4),
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
