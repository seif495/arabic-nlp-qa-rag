"""Vercel Python function: JSON API for the MS3 Arabic RAG chatbot.

All ``/api/*`` paths are rewritten to this function (see ``web/vercel.json``);
the static UI in ``web/public/`` calls ``/api/config`` and ``/api/chat``.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

WEB_ROOT = Path(__file__).resolve().parents[1]
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))

from ms3web.chat import (  # noqa: E402
    GENERATION_MODES,
    MEMORY_STRATEGIES,
    ChatSettings,
    RagEngine,
)
from ms3web.prompts import TEMPLATES  # noqa: E402

app = FastAPI(title="Arabic NLP RAG (MS3)", docs_url="/api/docs", openapi_url="/api/openapi.json")


@lru_cache(maxsize=1)
def get_engine() -> RagEngine:
    return RagEngine.from_assets()


class Turn(BaseModel):
    user: str = Field(max_length=4000)
    assistant: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=50)
    prompt_type: Literal["arabic_guided", "english_guided", "arabic_minimal", "english_minimal"] = "arabic_guided"
    memory_strategy: Literal["sliding_window", "full_history", "strict_truncation", "summarized_history"] = "sliding_window"
    generation_mode: Literal["llm_only", "ms2_only", "hybrid"] = "llm_only"
    use_reranking: bool = False
    top_k: int = Field(default=6, ge=1, le=15)
    ms2_model: Literal["model_a", "model_b", "transformer"] | None = None


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict:
    engine = get_engine()
    return {
        "prompt_types": list(TEMPLATES),
        "memory_strategies": list(MEMORY_STRATEGIES),
        "generation_modes": list(GENERATION_MODES),
        "llm_providers": engine.llm.providers,
        "ms2_available": engine.ms2_available,
        "ms2_default": engine.default_ms2,
        "ms2_models": [runtime.info for runtime in engine.ms2_models.values()],
        "chunks": len(engine.retriever.chunks),
        "episodes": sorted({chunk.episode for chunk in engine.retriever.chunks}),
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    engine = get_engine()
    result = engine.answer(
        request.question.strip(),
        [(turn.user, turn.assistant) for turn in request.history],
        ChatSettings(
            prompt_type=request.prompt_type,
            memory_strategy=request.memory_strategy,
            generation_mode=request.generation_mode,
            use_reranking=request.use_reranking,
            top_k=request.top_k,
            ms2_model=request.ms2_model,
        ),
    )
    return {
        "answer": result.answer,
        "llm_answer": result.llm_answer,
        "ms2_answer": result.ms2_answer,
        "standalone_query": result.standalone_query,
        "sources": result.sources,
        "logs": result.logs,
    }
