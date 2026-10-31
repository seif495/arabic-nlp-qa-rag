"""Groq (primary) -> Gemini (fallback) chat client over plain HTTPS.

Same provider strategy as ``src/ms3/generation/llm_manager.py`` but without
LangChain, keeping the Vercel function bundle small. Keys come from the
Vercel project's environment variables (``GROQ_API_KEY``, ``GOOGLE_API_KEY``).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class LLMUnavailableError(RuntimeError):
    """Raised when no provider is configured or every provider failed."""


@dataclass(frozen=True)
class LLMReply:
    text: str
    provider: str
    model: str


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
                    return LLMReply(self._groq(system, user), "groq", self.groq_model)
                return LLMReply(self._gemini(system, user), "gemini", self.gemini_model)
            except Exception as exc:  # noqa: BLE001 - fall through to next provider
                errors.append(f"{provider}: {exc}")
        if not errors:
            raise LLMUnavailableError("no LLM API key configured")
        raise LLMUnavailableError("; ".join(errors))

    # -- providers ----------------------------------------------------------

    def _groq(self, system: str, user: str) -> str:
        messages = [{"role": "user", "content": user}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        payload = {
            "model": self.groq_model,
            "messages": messages,
            "temperature": self.temperature,
        }
        data = self._post(
            GROQ_URL,
            payload,
            {"Authorization": f"Bearer {self.groq_api_key}"},
        )
        return data["choices"][0]["message"]["content"].strip()

    def _gemini(self, system: str, user: str) -> str:
        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        data = self._post(
            GEMINI_URL.format(model=self.gemini_model),
            payload,
            {"x-goog-api-key": self.google_api_key or ""},
        )
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts).strip()

    def _post(self, url: str, payload: dict, headers: dict[str, str]) -> dict:
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json", **headers},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = RuntimeError(f"HTTP {exc.code}")
                if exc.code not in RETRYABLE_STATUS:
                    break
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(0.5 * 2**attempt)
        raise last_error or RuntimeError("request failed")
