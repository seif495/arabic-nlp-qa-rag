from __future__ import annotations

import argparse
import math
from pathlib import Path

from src.common.paths import resolve_ms1_paths
from src.ms1.ingest.loader import load_qa_files


def _question_token_lengths() -> list[int]:
    paths = resolve_ms1_paths()
    qa_records = load_qa_files(paths)
    return [len(record.get("question", "").split()) for record in qa_records]


def _build_histogram(lengths: list[int], bins: int) -> tuple[list[int], list[int]]:
    min_val = min(lengths)
    max_val = max(lengths)

    if min_val == max_val:
        return [len(lengths)], [min_val, max_val + 1]

    distinct_lengths = max_val - min_val + 1
    effective_bins = max(1, min(bins, distinct_lengths))
    bin_width = math.ceil(distinct_lengths / effective_bins)

    edges = [min_val + i * bin_width for i in range(effective_bins)]
    edges.append(max_val + 1)
    counts = [0] * effective_bins

    for value in lengths:
        idx = min((value - min_val) // bin_width, effective_bins - 1)
        counts[idx] += 1

    return counts, edges


def _render_svg(counts: list[int], edges: list[int], output_path: Path) -> None:
    width = 980
    height = 560
    left = 80
    right = 30
    top = 50
    bottom = 85

    chart_w = width - left - right
    chart_h = height - top - bottom

    max_count = max(counts) if counts else 1
    bar_gap = 6
    bars = len(counts)
    bar_w = max((chart_w - bar_gap * (bars + 1)) / max(bars, 1), 6)

    parts: list[str] = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')

    parts.append(
        f'<text x="{width / 2}" y="26" text-anchor="middle" font-size="20" font-family="Arial">MS1 Question Token-Length Distribution</text>'
    )

    parts.append(
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#1f2937" stroke-width="2"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#1f2937" stroke-width="2"/>'
    )

    y_ticks = 5
    for i in range(y_ticks + 1):
        frac = i / y_ticks
        y = top + chart_h - frac * chart_h
        tick_value = int(round(max_count * frac))
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_w}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#374151">{tick_value}</text>'
        )

    for i, count in enumerate(counts):
        bar_h = (count / max_count) * chart_h if max_count else 0
        x = left + bar_gap + i * (bar_w + bar_gap)
        y = top + chart_h - bar_h
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#ff2d20" stroke="#b91c1c" stroke-width="1"/>'
        )

        start = edges[i]
        end = edges[i + 1] - 1
        x_label = f"{start}" if start == end else f"{start}-{end}"
        if bars <= 10 or i % math.ceil(bars / 8) == 0:
            parts.append(
                f'<text x="{x + bar_w / 2:.2f}" y="{top + chart_h + 22}" text-anchor="middle" font-size="11" font-family="Arial" fill="#374151">{x_label}</text>'
            )

    parts.append(
        f'<text x="{left + chart_w / 2}" y="{height - 30}" text-anchor="middle" font-size="14" font-family="Arial">Question Length (tokens)</text>'
    )
    parts.append(
        f'<text x="20" y="{top + chart_h / 2}" transform="rotate(-90 20,{top + chart_h / 2})" text-anchor="middle" font-size="14" font-family="Arial">Count</text>'
    )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def generate_question_histogram(output_path: Path, bins: int = 10) -> Path:
    lengths = _question_token_lengths()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts, edges = _build_histogram(lengths=lengths, bins=bins)
    _render_svg(counts=counts, edges=edges, output_path=output_path)
    return output_path


def main() -> None:
    default_output = (
        resolve_ms1_paths().experiments_ms1
        / "figures"
        / "ms1_question_token_length_histogram.svg"
    )

    parser = argparse.ArgumentParser(
        description="Generate a clean SVG histogram for MS1 question token lengths."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output SVG path (default: experiments/ms1/figures/ms1_question_token_length_histogram.svg)",
    )
    parser.add_argument("--bins", type=int, default=10, help="Histogram bin count")
    args = parser.parse_args()

    written = generate_question_histogram(output_path=args.output, bins=args.bins)
    print(f"Saved histogram: {written}")


if __name__ == "__main__":
    main()
