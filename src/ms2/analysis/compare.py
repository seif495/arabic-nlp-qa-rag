from __future__ import annotations

import csv
import json
from pathlib import Path


_SEEDS = (13, 42, 91)

# (display label, run_summary key, format_fn)
_METRIC_ROWS: tuple[tuple[str, str, str], ...] = (
    ("Exact Match — test",                "test_em",                          ".3f"),
    ("Token-F1 — test",                   "test_token_f1",                    ".3f"),
    ("Char edit distance — test",         "test_char_edit_distance",          ".3f"),
    ("BLEU-1 — test",                     "test_bleu1",                       ".3f"),
    ("Exact Match — dev",                 "dev_em",                           ".3f"),
    ("Token-F1 — dev",                    "dev_token_f1",                     ".3f"),
    ("Exact Match (leave-2-videos-out)",  None,                               ".3f"),
    ("Token-F1 (leave-2-videos-out)",     None,                               ".3f"),
    ("Parameter count",                   "parameter_count",                  ",d"),
    ("Training wall-clock (min)",         "train_wall_clock_minutes",         ".1f"),
    ("Inference time / example (ms)",     "mean_inference_time_ms_per_example", ".1f"),
    ("Peak GPU memory (MB)",              "peak_gpu_memory_mb",               ".0f"),
)


def write_headline_tables(
    experiments_dir: Path, reports_dir: Path
) -> tuple[Path, Path]:
    experiments_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summaries = _load_summaries(experiments_dir)

    rows = []
    for label, key, fmt in _METRIC_ROWS:
        model_a = _aggregate(summaries, "a", key, fmt)
        model_b = _aggregate(summaries, "b", key, fmt)
        status = "ok" if (model_a != "—" or model_b != "—") else "missing_runs"
        rows.append({"metric": label, "model_a": model_a, "model_b": model_b, "status": status})

    csv_path = experiments_dir / "headline_comparison_v001.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("metric", "model_a", "model_b", "status")
        )
        writer.writeheader()
        writer.writerows(rows)

    md_path = reports_dir / "ms2_headline_table.md"
    lines = [
        "# MS2 Headline Comparison",
        "",
        f"Seeds averaged: {', '.join(str(s) for s in _SEEDS)} (only completed runs included).",
        "",
        "| Metric | Model A | Model B |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {r['metric']} | {r['model_a']} | {r['model_b']} |" for r in rows
    )
    missing = [r["metric"] for r in rows if r["model_a"] == "—" and r["model_b"] == "—"]
    if missing:
        lines.append("")
        lines.append(
            "Cells marked `—` have no completed runs for either model."
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def _load_summaries(
    experiments_dir: Path,
) -> dict[tuple[str, int], dict]:
    summaries: dict[tuple[str, int], dict] = {}
    for model in ("a", "b"):
        for seed in _SEEDS:
            p = experiments_dir / f"model_{model}" / str(seed) / "run_summary_v001.json"
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if data.get("status") == "trained":
                        summaries[(model, seed)] = data
                except (json.JSONDecodeError, OSError):
                    pass
    return summaries


def _aggregate(
    summaries: dict[tuple[str, int], dict],
    model: str,
    key: str | None,
    fmt: str,
) -> str:
    if key is None:
        return "—"
    vals = [
        s[key]
        for (m, _), s in summaries.items()
        if m == model and isinstance(s.get(key), (int, float))
    ]
    if not vals:
        return "—"
    avg = sum(vals) / len(vals)
    return format(int(avg) if fmt.endswith("d") else avg, fmt)
