"""Groq (primary) -> Gemini (fallback) chat client over plain HTTPS.

Same provider strategy as ``src/ms3/generation/llm_manager.py`` but without
LangChain, keeping the Vercel function bundle small. Keys come from the
Vercel project's environment variables (``GROQ_API_KEY``, ``GOOGLE_API_KEY``).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# llama-3.3-70b-versatile was shut down on 2026-08-16; these are Groq's
# recommended replacements (console.groq.com/docs/deprecations), in order.
PREFERRED_GROQ_MODELS = ("openai/gpt-oss-120b", "qwen/qwen3.6-27b", "openai/gpt-oss-20b")
DEFAULT_GROQ_MODEL = PREFERRED_GROQ_MODELS[0]
NON_CHAT_MODEL_MARKERS = ("whisper", "tts", "guard", "orpheus", "playai", "compound", "embed")
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
# Groq sits behind Cloudflare, which rejects urllib's default
# "Python-urllib/3.x" agent with "403 error code: 1010" before auth.
USER_AGENT = "arabizy-ms3/1.0 (+https://github.com/seif495/arabic-nlp-qa-rag)"
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMUnavailableError(RuntimeError):
    """Raised when no provider is configured or every provider failed."""


class HTTPStatusError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


@dataclass(frozen=True)
class LLMReply:
    text: str
    provider: str
    model: str
    # Errors from providers tried (and failed) before this one answered.
    fallback_errors: tuple[str, ...] = ()


class LLMClient:
    def __init__(
        self,
        groq_api_key: str | None = None,
        google_api_key: str | None = None,
        groq_model: str | None = None,
        gemini_model: str | None = None,
        timeout: float = 25.0,
        max_retries: int = 2,
        temperature: float = 0.1,
    ) -> None:
        self.groq_api_key = groq_api_key
        self.google_api_key = google_api_key
        self.groq_model = groq_model or DEFAULT_GROQ_MODEL
        self.gemini_model = gemini_model or DEFAULT_GEMINI_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature

    @classmethod
    def from_env(cls) -> LLMClient:
        return cls(
            groq_api_key=os.environ.get("GROQ_API_KEY") or None,
            google_api_key=os.environ.get("GOOGLE_API_KEY") or None,
            groq_model=os.environ.get("GROQ_MODEL") or None,
            gemini_model=os.environ.get("GEMINI_MODEL") or None,
        )

    @property
    def providers(self) -> list[str]:
        names = []
        if self.groq_api_key:
            names.append("groq")
        if self.google_api_key:
            names.append("gemini")
        return names

    @property
    def available(self) -> bool:
        return bool(self.providers)

    def complete(self, system: str, user: str) -> LLMReply:
        errors: list[str] = []
        for provider in self.providers:
            try:
                if provider == "groq":
                    return LLMReply(self._groq(system, user), "groq", self.groq_model, tuple(errors))
                return LLMReply(self._gemini(system, user), "gemini", self.gemini_model, tuple(errors))
            except Exception as exc:  # noqa: BLE001 - fall through to next provider
                errors.append(f"{provider}: {exc}")
        if not errors:
            raise LLMUnavailableError("no LLM API key configured")
        raise LLMUnavailableError("; ".join(errors))

    # -- providers ----------------------------------------------------------

    def _groq(self, system: str, user: str) -> str:
        try:
            return self._groq_chat(system, user)
        except HTTPStatusError as exc:
            # Groq retires models (llama-3.3-70b-versatile went away in 2026);
            # switch to one this key can use instead of failing every call.
            if exc.status != 404 or "model_not_found" not in exc.body:
                raise
            replacement = self._discover_groq_model()
            if replacement is None or replacement == self.groq_model:
                raise
            self.groq_model = replacement
            return self._groq_chat(system, user)

    def _groq_chat(self, system: str, user: str) -> str:
        messages = [{"role": "user", "content": user}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        payload: dict = {
            "model": self.groq_model,
            "messages": messages,
            "temperature": self.temperature,
        }
        # Reasoning models: keep latency low and the reasoning out of the answer.
        if self.groq_model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = "low"
        elif self.groq_model.startswith("qwen/"):
            payload["reasoning_format"] = "hidden"
        data = self._request(GROQ_URL, payload, self._groq_headers())
        content = data["choices"][0]["message"].get("content") or ""
        return _THINK_BLOCK.sub("", content).strip()

    def _groq_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.groq_api_key}"}

    def _discover_groq_model(self) -> str | None:
        """Pick a chat model this key can use from ``GET /openai/v1/models``."""
        data = self._request(GROQ_MODELS_URL, None, self._groq_headers())
        available = [m["id"] for m in data.get("data", []) if m.get("active", True)]
        for model in PREFERRED_GROQ_MODELS:
            if model in available:
                return model
        chat = [m for m in available if not any(marker in m.lower() for marker in NON_CHAT_MODEL_MARKERS)]
        return chat[0] if chat else None

    def _gemini(self, system: str, user: str) -> str:
        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        data = self._request(
            GEMINI_URL.format(model=self.gemini_model),
            payload,
            {"x-goog-api-key": self.google_api_key or ""},
        )
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts).strip()

    def _request(self, url: str, payload: dict | None, headers: dict[str, str]) -> dict:
        """POST ``payload`` as JSON, or GET when it is ``None``; retry transient errors."""
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                    **headers,
                },
                method="GET" if body is None else "POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                last_error = HTTPStatusError(exc.code, detail)
                if exc.code not in RETRYABLE_STATUS:
                    break
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(0.5 * 2**attempt)
        raise last_error or RuntimeError("request failed")
