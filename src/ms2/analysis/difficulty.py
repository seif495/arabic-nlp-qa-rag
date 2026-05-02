from __future__ import annotations

import json
from pathlib import Path


DIFFICULTY_BUCKETS = ("easy", "medium", "hard")


def normalize_difficulty(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in DIFFICULTY_BUCKETS:
        return lowered
    if lowered in {"سهل", "easy"}:
        return "easy"
    if lowered in {"متوسط", "medium"}:
        return "medium"
    return "hard"


def write_difficulty_placeholder(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "missing_trained_checkpoint", "buckets": {bucket: None for bucket in DIFFICULTY_BUCKETS}}, indent=2) + "\n", encoding="utf-8")
    return path
