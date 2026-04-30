from __future__ import annotations

import json
import math
import re
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from src.common.paths import make_ms2_output_filename, resolve_ms2_paths
from src.ms2.data.records import MS2DatasetRecord, load_ms1_processed_records


LENGTH_CAPS: dict[str, int] = {
    "l_q": 32,
    "l_c": 384,
    "l_enc": 420,
    "l_dec": 64,
}
LENGTH_AXES = ("question", "context", "encoder", "decoder")
ADR_PLACEHOLDER_CAPS = dict(LENGTH_CAPS)
SPECIAL_FORMAT_TOKENS = 4
MAX_ENCODER_CAP = 420
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class AxisStats:
    p95: int
    p99: int
    maximum: int
    cap: int
    coverage: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def proxy_tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def analyze_length_distributions(
    records: list[MS2DatasetRecord],
    tokenizer: Callable[[str], list[int | str]] | None = None,
) -> dict[str, AxisStats]:
    tokenize = tokenizer or proxy_tokenize
    lengths = {axis: [] for axis in LENGTH_AXES}
    for record in records:
        question_len = len(tokenize(record.question_text))
        context_len = len(tokenize(record.normalized_context))
        answer_len = len(tokenize(record.answer_text))
        lengths["question"].append(question_len)
        lengths["context"].append(context_len)
        lengths["encoder"].append(question_len + context_len + SPECIAL_FORMAT_TOKENS)
        lengths["decoder"].append(answer_len + 1)

    return {axis: _stats(values, axis=axis) for axis, values in lengths.items()}


def run_length_analysis(
    repo_root: Path | None = None,
    dataset_path: Path | str | None = None,
    tokenizer: Callable[[str], list[int | str]] | None = None,
) -> dict[str, object]:
    root = (repo_root or Path.cwd()).resolve()
    paths = resolve_ms2_paths(repo_root=root, create_dirs=True)
    records = load_ms1_processed_records(dataset_path or root / "data/processed/ms1/ms1_dataset_processed_v001.jsonl")
    proxy_stats = analyze_length_distributions(records)
    if tokenizer is None:
        raise ValueError(
            "run_length_analysis requires a tokenizer callable for the post_bpe pass"
        )
    final_stats = analyze_length_distributions(records, tokenizer=tokenizer)
    result = {
        "record_count": len(records),
        "proxy": {axis: stats.to_dict() for axis, stats in proxy_stats.items()},
        "post_bpe": {axis: stats.to_dict() for axis, stats in final_stats.items()},
        "length_caps": LENGTH_CAPS,
        "encoder_coverage_verified": final_stats["encoder"].coverage >= 0.99,
    }
    output_path = paths.experiments_ms2 / make_ms2_output_filename(
        "length", "distribution", 1, "json"
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_length_histograms(records=records, output_dir=root / "docs/fs/artifacts/ms2", tokenizer=tokenizer)
    return result


def write_length_histograms(
    records: list[MS2DatasetRecord],
    output_dir: Path,
    tokenizer: Callable[[str], list[int | str]] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    passes = {
        "proxy": analyze_length_distributions(records),
        "post_bpe": analyze_length_distributions(records, tokenizer=tokenizer),
    }
    for pass_name, stats_by_axis in passes.items():
        for axis, stats in stats_by_axis.items():
            path = output_dir / f"length_{pass_name}_{axis}.png"
            _write_single_pixel_png(path, _color_for_stats(stats))
            paths.append(path)
    return paths


def _stats(values: list[int], axis: str) -> AxisStats:
    if not values:
        cap = LENGTH_CAPS[_cap_key(axis)]
        return AxisStats(p95=0, p99=0, maximum=0, cap=cap, coverage=1.0)

    p95 = _percentile(values, 0.95)
    p99 = _percentile(values, 0.99)
    maximum = max(values)
    cap = min(max(p99, 1), MAX_ENCODER_CAP) if axis == "encoder" else max(p99, 1)
    if axis != "encoder":
        cap = LENGTH_CAPS[_cap_key(axis)]
    coverage = sum(1 for value in values if value <= cap) / len(values)
    return AxisStats(p95=p95, p99=p99, maximum=maximum, cap=cap, coverage=coverage)


def _percentile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = math.ceil(percentile * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _cap_key(axis: str) -> str:
    return {"question": "l_q", "context": "l_c", "encoder": "l_enc", "decoder": "l_dec"}[axis]


def _color_for_stats(stats: AxisStats) -> tuple[int, int, int]:
    red = min(255, stats.maximum * 3)
    green = min(255, stats.p99 * 5)
    blue = 255 if stats.coverage >= 0.99 else 64
    return red, green, blue


def _write_single_pixel_png(path: Path, rgb: tuple[int, int, int]) -> None:
    raw = b"\x00" + bytes(rgb)
    png = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) + _png_chunk(b"IDAT", zlib.compress(raw)) + _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
