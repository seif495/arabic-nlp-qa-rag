from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ms3web import chat as chat_module  # noqa: E402
from ms3web.chat import ChatSettings, RagEngine, format_history  # noqa: E402
from ms3web.ms2 import TransformerRuntime  # noqa: E402
from ms3web.llm import LLMClient, LLMReply, LLMUnavailableError  # noqa: E402
from ms3web.seq2seq import BOS_ID, Seq2SeqTransformer  # noqa: E402
from ms3web.retrieval import BM25Retriever, Chunk, tokenize  # noqa: E402

CHUNKS = [
    Chunk(0, "تاج محل", "شاه جهان بنى تاج محل في أجرا عشان مراته ممتاز محل."),
    Chunk(1, "الأخطبوط", "الأخطبوط عنده تلات قلوب ودمه لونه أزرق."),
    Chunk(2, "منابع النيل", "نهر النيل بيبدأ من بحيرة فيكتوريا والهضبة الإثيوبية."),
]


class FakeLLM(LLMClient):
    def __init__(self, answer: str = "شاه جهان", fail: bool = False) -> None:
        super().__init__(groq_api_key="test")
        self.answer = answer
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> LLMReply:
        self.calls.append((system, user))
        if self.fail:
            raise LLMUnavailableError("boom")
        if "Standalone Query" in user:
            return LLMReply("مين بنى تاج محل", "groq", "fake")
        return LLMReply(self.answer, "groq", "fake")


def tiny_model_b() -> Seq2SeqTransformer:
    rng = np.random.default_rng(0)
    d, ff, vocab = 96, 64, 64
    w: dict[str, np.ndarray] = {
        "embedding": rng.normal(0, 0.1, (vocab, d)),
        "output_bias": np.zeros(vocab),
    }

    def dense(name: str, i: int, o: int) -> None:
        w[f"{name}.kernel"] = rng.normal(0, 0.1, (i, o))
        w[f"{name}.bias"] = np.zeros(o)

    def ln(name: str) -> None:
        w[f"{name}.gamma"], w[f"{name}.beta"] = np.ones(d), np.zeros(d)

    def attn(name: str) -> None:
        for proj in ("wq", "wk", "wv", "wo"):
            dense(f"{name}.{proj}", d, d)

    ln("encoder.0.ln_self"), attn("encoder.0.self_attn"), ln("encoder.0.ln_ffn")
    dense("encoder.0.ffn.inner", d, ff), dense("encoder.0.ffn.outer", ff, d), ln("encoder.final_ln")
    for part in ("ln_self", "ln_cross", "ln_ffn"):
        ln(f"decoder.0.{part}")
    attn("decoder.0.self_attn"), attn("decoder.0.cross_attn")
    dense("decoder.0.ffn.inner", d, ff), dense("decoder.0.ffn.outer", ff, d), ln("decoder.final_ln")
    meta = dict(d_model=d, n_heads=2, n_encoder_layers=1, n_decoder_layers=1, ln_epsilon=1e-3,
                positional_mode=1, rope_theta_base=10000.0, encoder_max_length=420, decoder_max_length=64)
    w.update({f"meta.{k}": np.asarray(v) for k, v in meta.items()})
    return Seq2SeqTransformer(w)


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [5 + (ord(ch) % 50) for ch in text if not ch.isspace()][:40]

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)

    def encoder_input(self, question: str, context_ids: list[int]) -> list[int]:
        ids = [BOS_ID, *self.encode(question), 4, *context_ids, 3]
        return ids + [0] * (64 - len(ids))

    def context_windows(self, text: str, max_windows: int = 4) -> list[list[int]]:
        return [self.encode(text)]


TINY_INFO = {"model": "model_b", "display_name": "Model B (RoPE Transformer)", "dev_token_f1": 0.0, "dev_em": 0.0}


def make_engine(llm: LLMClient | None = None, with_ms2: bool = True) -> RagEngine:
    return RagEngine(
        BM25Retriever(CHUNKS),
        {"model_b": TransformerRuntime(tiny_model_b(), FakeTokenizer(), TINY_INFO)} if with_ms2 else {},
        "model_b" if with_ms2 else None,
        llm or FakeLLM(),
    )


def test_tokenize_normalises_and_light_stems() -> None:
    assert tokenize("الأخطبوط") == tokenize("اخطبوط")
    assert "في" not in tokenize("في تاج محل")


def test_bm25_ranks_topical_chunk_first() -> None:
    results = BM25Retriever(CHUNKS).search("الاخطبوط عنده كام قلب", k=2)
    assert results[0].chunk.episode == "الأخطبوط"


def test_llm_only_turn_uses_standalone_query_and_prompt_strategy() -> None:
    llm = FakeLLM()
    result = make_engine(llm).answer("مين بناه؟", [("تاج محل فين", "في أجرا")], ChatSettings(prompt_type="english_guided"))
    assert result.answer == "شاه جهان"
    assert result.standalone_query == "مين بنى تاج محل"
    assert result.sources[0]["episode"] == "تاج محل"
    system, human = llm.calls[-1]
    assert system.startswith("You are a helpful AI assistant")
    assert "شاه جهان بنى تاج محل" in system  # retrieved context injected
    assert "User: تاج محل فين" in human  # sliding-window history injected


def test_out_of_domain_is_replaced_by_rejection_message() -> None:
    result = make_engine(FakeLLM(answer="OUT_OF_DOMAIN")).answer("ما عاصمة فرنسا؟", [], ChatSettings())
    assert result.answer == chat_module.OUT_OF_DOMAIN_MESSAGE


def test_hybrid_mode_runs_model_b_rope_with_reranking() -> None:
    result = make_engine().answer(
        "مين بنى تاج محل", [], ChatSettings(generation_mode="hybrid", use_reranking=True, top_k=2)
    )
    assert result.llm_answer == "شاه جهان"
    assert result.ms2_answer
    assert result.logs["reranking"] is True
    assert all(source["ms2_score"] is not None for source in result.sources)


def test_ms2_only_mode_works_without_llm_keys() -> None:
    engine = make_engine(LLMClient())
    result = engine.answer("تاج محل", [], ChatSettings(generation_mode="ms2_only"))
    assert result.answer == result.ms2_answer
    assert result.llm_answer is None


def test_llm_failure_never_crashes() -> None:
    result = make_engine(FakeLLM(fail=True)).answer("تاج محل", [], ChatSettings())
    assert result.answer == chat_module.LLM_ERROR_MESSAGE
    assert result.logs["llm_errors"]


def test_missing_model_b_falls_back_to_llm_only() -> None:
    result = make_engine(with_ms2=False).answer("تاج محل", [], ChatSettings(generation_mode="ms2_only"))
    assert result.logs["generation_mode"] == "llm_only"


def test_greedy_decode_stops_degenerate_repetition() -> None:
    model = tiny_model_b()
    # Force the same token every step by biasing the output layer.
    model.output_bias = np.zeros_like(model.output_bias)
    model.output_bias[7] = 1e4
    assert model.greedy_decode([BOS_ID, 9, 4, 11, 3], max_repeat=3).token_ids == [7, 7, 7]
    assert len(model.greedy_decode([BOS_ID, 9, 4, 11, 3], max_repeat=None).token_ids) == 64


@pytest.mark.parametrize(
    ("strategy", "expected_turns"),
    [("sliding_window", 3), ("full_history", 5), ("strict_truncation", 5)],
)
def test_memory_strategies(strategy: str, expected_turns: int) -> None:
    history = [(f"q{i}", f"a{i}") for i in range(5)]
    assert format_history(history, strategy).count("User:") == expected_turns


def test_api_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.index as api_index

    api_index.get_engine.cache_clear()
    monkeypatch.setattr(api_index, "get_engine", lambda: make_engine())
    client = TestClient(api_index.app)

    config = client.get("/api/config").json()
    assert config["ms2_available"] is True
    assert config["ms2_default"] == "model_b"
    assert [m["model"] for m in config["ms2_models"]] == ["model_b"]

    response = client.post(
        "/api/chat",
        json={"question": "مين بنى تاج محل", "generation_mode": "hybrid", "history": [{"user": "a", "assistant": "b"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["llm_answer"] == "شاه جهان" and body["ms2_answer"]
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_api_works_when_loaded_from_file_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serverless runtimes may exec the entrypoint without registering it in sys.modules."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "web/api/index.py"
    spec = importlib.util.spec_from_file_location("vercel_entrypoint_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_engine", lambda: make_engine())
    client = TestClient(module.app)
    response = client.post(
        "/api/chat", json={"question": "مين بنى تاج محل", "history": [{"user": "a", "assistant": "b"}]}
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "شاه جهان"


@pytest.mark.parametrize(
    ("method", "url", "route"),
    [
        ("GET", "/api/index?op=config", "config"),  # what the UI calls
        ("GET", "/api/index?op=index&op=config", "config"),  # rewrite + original query merged
        ("GET", "/api/config", "config"),  # pretty URL, path preserved
        ("GET", "/config", "config"),  # /api prefix stripped
        ("GET", "/api/index", "health"),  # bare function path
        ("GET", "/api/health", "health"),
        ("GET", "/health", "health"),
        ("POST", "/api/index?op=chat", "chat"),
        ("POST", "/api/chat", "chat"),
        ("POST", "/chat", "chat"),
    ],
)
def test_routes_survive_any_vercel_path(monkeypatch: pytest.MonkeyPatch, method: str, url: str, route: str) -> None:
    import api.index as api_index

    monkeypatch.setattr(api_index, "get_engine", lambda: make_engine())
    client = TestClient(api_index.app)
    if method == "POST":
        response = client.post(url, json={"question": "مين بنى تاج محل"})
    else:
        response = client.get(url)
    assert response.status_code == 200, response.text
    body = response.json()
    expected_key = {"health": "status", "config": "ms2_models", "chat": "answer"}[route]
    assert expected_key in body


def test_router_leaves_docs_and_root_alone() -> None:
    from api.index import resolve_route

    assert resolve_route("/", b"") is None
    assert resolve_route("/api/docs", b"") is None
    assert resolve_route("/api/openapi.json", b"") is None
    assert resolve_route("/app.js", b"") is None


def test_health_reports_path_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.index as api_index

    client = TestClient(api_index.app)
    assert client.get("/api/index").json() == {"status": "ok", "path_seen": "/api/index"}


def test_fallback_provider_errors_are_logged() -> None:
    class GroqDownLLM(LLMClient):
        def _groq(self, system: str, user: str) -> str:
            raise RuntimeError("HTTP 401: invalid api key")

        def _gemini(self, system: str, user: str) -> str:
            return "مين بنى تاج محل" if "Standalone Query" in user else "شاه جهان"

    llm = GroqDownLLM(groq_api_key="g", google_api_key="k")
    result = make_engine(llm).answer("مين بنى تاج محل", [], ChatSettings())
    assert result.answer == "شاه جهان"
    assert result.logs["llm_provider"].startswith("gemini:")
    assert result.logs["llm_errors"] == ["groq: HTTP 401: invalid api key"] * 2


def test_ui_calls_pretty_api_urls_matching_vercel_rewrite() -> None:
    """Verified on the live deployment: Vercel replaces the query of
    ``/api/index?op=...`` with the rewrite's ``op=index``, so the UI must call
    ``/api/<route>`` and let ``/api/:op -> /api/index?op=:op`` route it."""
    import json
    from pathlib import Path

    web = Path(__file__).resolve().parents[2] / "web"
    app_js = (web / "public/app.js").read_text(encoding="utf-8")
    assert 'fetch("/api/config")' in app_js
    assert 'fetch("/api/chat"' in app_js
    assert "/api/index?op=" not in app_js
    rewrites = json.loads((web / "vercel.json").read_text(encoding="utf-8"))["rewrites"]
    assert {"source": "/api/:op", "destination": "/api/index?op=:op"} in rewrites


def test_llm_requests_send_explicit_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live Groq answered "403 error code: 1010" (Cloudflare blocks Python-urllib's agent)."""
    import io
    import json
    import urllib.request

    from ms3web import llm as llm_module

    seen = []

    def fake_urlopen(request, timeout):
        seen.append(request)
        payload = {"choices": [{"message": {"content": "ok"}}]}
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    reply = LLMClient(groq_api_key="k").complete("sys", "hi")
    assert reply.text == "ok" and reply.provider == "groq"
    agent = seen[0].get_header("User-agent")
    assert agent == llm_module.USER_AGENT
    assert not agent.startswith("Python-urllib")
    assert seen[0].get_header("Authorization") == "Bearer k"
