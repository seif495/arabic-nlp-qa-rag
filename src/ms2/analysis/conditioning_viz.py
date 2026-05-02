from __future__ import annotations

import struct
import zlib
from pathlib import Path


def write_placeholder_png(path: Path) -> Path:
    """Write a tiny valid PNG placeholder for missing trained-checkpoint visualizations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 8, 8
    raw = b"".join(b"\x00" + bytes([220, 226, 235]) * width for _ in range(height))
    png = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b"")
    path.write_bytes(png)
    return path


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
