"""Vendored copies in web/ must not drift from their src/ originals."""

from __future__ import annotations

from pathlib import Path

from ms3web import prompts as web_prompts
from src.ms3.generation import prompts as ms3_prompts

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_normalizer_is_verbatim_copy_of_ms2_metrics_normalize() -> None:
    original = (REPO_ROOT / "src/ms2/metrics/normalize.py").read_text(encoding="utf-8")
    vendored = (REPO_ROOT / "web/ms3web/normalize.py").read_text(encoding="utf-8")
    assert vendored == original


def test_answer_prompts_match_ms3_prompt_strategies() -> None:
    for name in ("ARABIC_SYSTEM_GUIDED", "ENGLISH_SYSTEM_GUIDED", "ARABIC_MINIMAL", "ENGLISH_MINIMAL"):
        assert getattr(web_prompts, name) == getattr(ms3_prompts, name), name
    human = ms3_prompts.get_prompt("english_guided").messages[1].prompt.template
    assert web_prompts.HUMAN_TEMPLATE == human


def test_web_requirements_exclude_heavy_frameworks() -> None:
    lines = (REPO_ROOT / "web/requirements.txt").read_text(encoding="utf-8").lower().splitlines()
    requirements = "\n".join(line for line in lines if not line.startswith("#"))
    for heavy in ("tensorflow", "torch", "sentence-transformers", "chromadb", "langchain", "streamlit"):
        assert heavy not in requirements
