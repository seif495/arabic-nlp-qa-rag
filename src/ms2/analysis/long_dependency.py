from __future__ import annotations

import json
from pathlib import Path


def question_answer_distance(
    context_tokens: list[str], question_tokens: list[str], answer_start: int
) -> int:
    """Distance from nearest question-token match in context to answer start."""
    content = [token for token in question_tokens if len(token) > 1]
    matches = [idx for idx, token in enumerate(context_tokens) if token in content]
    if not matches:
        return abs(answer_start)
    return min(abs(idx - answer_start) for idx in matches)


def write_long_dependency_placeholder(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": "missing_trained_checkpoint", "bins": []}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return path
