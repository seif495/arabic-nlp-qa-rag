"""Vercel Python function: JSON API for the MS3 Arabic RAG chatbot.

Routes: ``/api/health``, ``/api/config``, ``/api/chat``. Vercel can hand this
app a different path than the browser requested: the ``/api/index`` rewrite
target, the bare function path, or ``/`` (classic ``api/`` functions vs. the
FastAPI framework preset). So the UI calls the function's own path with the
route in the query (``/api/index?op=config``), and :class:`RouteFromRequest`
maps whatever path arrives back onto the right route.
"""

# No ``from __future__ import annotations`` here: pydantic must resolve the
# request models' annotations at class creation, because Vercel may load
# this file without registering it in ``sys.modules`` (string annotations
# would then fail with "ChatRequest is not fully defined" on /api/chat).
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Receive, Scope, Send

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

ROUTES = ("health", "config", "chat")
ORIGINAL_PATH_KEY = "ms3_original_path"


def resolve_route(path: str, query_string: bytes) -> str | None:
    """Route name for a request, from ``?op=`` first, else the last path segment."""
    for op in parse_qs(query_string.decode("latin-1")).get("op", []):
        if op in ROUTES:
            return op
    tail = path.rstrip("/").rsplit("/", 1)[-1]
    if tail in ROUTES:
        return tail
    # The bare function path; never "/" itself, which is the UI's index.html.
    if tail in ("api", "index", "index.py"):
        return "health"
    return None


class RouteFromRequest:
    """Rewrite the ASGI path to ``/api/<route>`` before FastAPI routes it."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            route = resolve_route(scope["path"], scope.get("query_string", b""))
            if route is not None:
                path = f"/api/{route}"
                scope = {**scope, "path": path, "raw_path": path.encode(), ORIGINAL_PATH_KEY: scope["path"]}
        await self.app(scope, receive, send)


app = FastAPI(title="Arabic NLP RAG (MS3)", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(RouteFromRequest)


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
def health(request: Request) -> dict:
    # ``path_seen`` shows what Vercel actually handed the app (for debugging routing).
    return {"status": "ok", "path_seen": request.scope.get(ORIGINAL_PATH_KEY, request.url.path)}


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
