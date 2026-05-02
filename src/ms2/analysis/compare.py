from __future__ import annotations

import csv
from pathlib import Path


ROWS = (
    "Exact Match (random split)",
    "Token-F1 (random split)",
    "Char edit distance (random split)",
    "BLEU-1 (random split)",
    "Exact Match (leave-2-videos-out)",
    "Token-F1 (leave-2-videos-out)",
    "Parameter count",
    "Training wall-clock",
    "Inference time / example (greedy)",
    "Peak GPU memory (training)",
)


def write_headline_tables(
    experiments_dir: Path, reports_dir: Path
) -> tuple[Path, Path]:
    experiments_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = experiments_dir / "headline_comparison_v001.csv"
    md_path = reports_dir / "ms2_headline_table.md"
    rows = []
    for row in ROWS:
        model_a = "~2.7M" if row == "Parameter count" else "—"
        model_b = "~3.8M" if row == "Parameter count" else "—"
        if row == "Training wall-clock":
            model_a = model_b = "target 75 min; not run locally"
        rows.append(
            {
                "metric": row,
                "model_a": model_a,
                "model_b": model_b,
                "status": "compute_bound_missing_runs",
            }
        )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("metric", "model_a", "model_b", "status")
        )
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# MS2 Headline Table",
        "",
        "| Metric | Model A | Model B |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['metric']} | {row['model_a']} | {row['model_b']} |" for row in rows
    )
    lines.append(
        "\nCells marked `—` require trained checkpoints; no scores were fabricated."
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path
