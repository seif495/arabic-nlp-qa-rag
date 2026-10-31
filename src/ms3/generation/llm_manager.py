"""
MS3 LLM Manager
===============
Manages LLM instances with API-failure handling, retry mechanisms,
and a primary → fallback strategy using free-tier models.

Primary  : Groq  → openai/gpt-oss-120b  (GROQ_API_KEY)
Fallback : Gemini → gemini-2.5-flash     (GOOGLE_API_KEY)

llama-3.3-70b-versatile (Groq, shut down 2026-08-16) and gemini-1.5-flash
were retired; these are their current replacements.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

_GROQ_MODEL = "openai/gpt-oss-120b"
_GEMINI_MODEL = "gemini-2.5-flash"


class LLMManager:
    """
    Returns a robust LangChain LLM with:
    - Automatic retries on the primary model (Groq).
    - Transparent fallback to Gemini if Groq is unavailable.
    - Never raises — callers always get a BaseChatModel back.
    """

    def __init__(
        self,
        primary_model: str = "groq",
        fallback_model: str = "gemini",
    ) -> None:
        self.primary_model_type = primary_model
        self.fallback_model_type = fallback_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_llm(self) -> BaseChatModel:
        """Return a robust LLM (primary with fallback)."""
        primary = self._init_model(self.primary_model_type)
        fallback = self._init_model(self.fallback_model_type)
        return primary.with_fallbacks([fallback])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_model(self, model_type: str) -> BaseChatModel:
        if model_type == "groq":
            return self._make_groq()
        if model_type == "gemini":
            return self._make_gemini()
        # Last-resort: try Groq
        return self._make_groq()

    @staticmethod
    def _make_groq() -> BaseChatModel:
        """Initialise ChatGroq (openai/gpt-oss-120b, free tier)."""
        from langchain_groq import ChatGroq

        api_key: Optional[str] = os.environ.get("GROQ_API_KEY")
        return ChatGroq(
            model=_GROQ_MODEL,
            api_key=api_key or "missing_key",
            temperature=0.1,
            max_retries=3,
        )

    @staticmethod
    def _make_gemini() -> BaseChatModel:
        """Initialise ChatGoogleGenerativeAI (gemini-2.5-flash, free tier)."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key: Optional[str] = os.environ.get("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=_GEMINI_MODEL,
            google_api_key=api_key or "missing_key",
            temperature=0.1,
            max_retries=3,
        )
