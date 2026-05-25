from __future__ import annotations

import csv
import random
from pathlib import Path

NOISE_TYPES = (
    "char_swap",
    "char_delete",
    "char_insert",
    "token_mask",
    "diacritic_inject",
)
NOISE_RATES = (0.0, 0.05, 0.1, 0.2, 0.4)
DIACRITICS = ("َ", "ُ", "ِ", "ْ", "ّ", "ً", "ٌ", "ٍ")


def apply_noise(text: str, noise_type: str, rate: float, seed: int = 13) -> str:
    rng = random.Random(seed)
    chars = list(text)
    if rate <= 0:
        return text
    if noise_type == "char_swap":
        for idx in range(len(chars) - 1):
            if rng.random() < rate:
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
    if noise_type == "char_delete":
        return "".join(ch for ch in chars if rng.random() >= rate)
    if noise_type == "char_insert":
        alphabet = chars or ["ا"]
        out: list[str] = []
        for ch in chars:
            out.append(ch)
            if rng.random() < rate:
                out.append(rng.choice(alphabet))
        return "".join(out)
    if noise_type == "token_mask":
        return " ".join(
            "<unk>" if rng.random() < rate else token for token in text.split()
        )
    if noise_type == "diacritic_inject":
        out = []
        for ch in chars:
            out.append(ch)
            if rng.random() < rate:
                out.append(rng.choice(DIACRITICS))
        return "".join(out)
    raise ValueError(f"unknown noise type: {noise_type}")


def write_noise_battery_placeholder(path: Path) -> Path:
    """Write tidy placeholder rows without fabricated scores."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "seed", "noise_type", "rate", "token_f1", "status"),
        )
        writer.writeheader()
        for model in ("A", "B"):
            for seed in (13, 42, 91):
                for noise_type in NOISE_TYPES:
                    for rate in NOISE_RATES:
                        writer.writerow(
                            {
                                "model": model,
                                "seed": seed,
                                "noise_type": noise_type,
                                "rate": rate,
                                "token_f1": "",
                                "status": "missing_trained_checkpoint",
                            }
                        )
    return path
