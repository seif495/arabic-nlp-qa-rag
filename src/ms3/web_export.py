"""Build the static assets the Vercel deployment (``web/``) ships with.

``python -m src.cli.ms3 export-web`` writes into ``web/ms3web/assets/``:

- ``ms3_chunks.json``: MS1-normalised transcripts chunked exactly like the
  MS3 vector store (``DocumentLoader(chunk_size=1000, chunk_overlap=200)``);
- ``ms2_tokenizer.model``: the MS2 SentencePiece model Model B was trained with;
- ``ms2_model_b_rope.npz``: trained Model B (RoPE Transformer) weights for the
  NumPy runtime in ``web/ms3web/model_b.py``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.common.paths import resolve_ms2_paths

WEB_ASSETS_RELATIVE = Path("web/ms3web/assets")
CHUNKS_FILENAME = "ms3_chunks.json"
TOKENIZER_FILENAME = "ms2_tokenizer.model"
MODEL_B_FILENAME = "ms2_model_b_rope.npz"
DEFAULT_MODEL_B_CHECKPOINT = Path("experiments/ms2/model_b/42/checkpoints/best.weights.h5")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class WebExportResult:
    chunks_path: Path
    chunk_count: int
    tokenizer_path: Path | None
    model_b_path: Path | None


def episode_title(filename: str) -> str:
    """``"تاج محل  الدحيح_normalized.txt"`` -> ``"تاج محل"``."""
    stem = Path(filename).stem.removesuffix("_normalized")
    return stem.replace("الدحيح", "").strip()


def export_chunks(normalized_dir: Path, output_path: Path) -> int:
    from src.ms3.retrieval.document_loader import DocumentLoader

    files = sorted(str(path) for path in normalized_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No normalized transcripts in {normalized_dir}; run `python -m src.cli.ms1 normalize` first"
        )
    loader = DocumentLoader(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = loader.load_and_chunk_episodes(files)
    payload = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunks": [
            {
                "chunk_id": idx,
                "episode": episode_title(doc.metadata["source_episode"]),
                "text": doc.page_content,
            }
            for idx, doc in enumerate(chunks)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=0) + "\n", encoding="utf-8"
    )
    return len(chunks)


def export_web_assets(
    repo_root: Path,
    model_b_checkpoint: Path | None = DEFAULT_MODEL_B_CHECKPOINT,
    normalized_dir: Path = Path("data/interim/normalized"),
) -> WebExportResult:
    repo_root = repo_root.resolve()
    assets = repo_root / WEB_ASSETS_RELATIVE
    chunks_path = assets / CHUNKS_FILENAME
    count = export_chunks(repo_root / normalized_dir, chunks_path)

    tokenizer_out = model_out = None
    if model_b_checkpoint is not None:
        checkpoint = repo_root / model_b_checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Model B checkpoint not found: {checkpoint}; train it with "
                "`python -m src.cli.ms2 train --model b ...` or pass --skip-model-b"
            )
        tokenizer_src = resolve_ms2_paths(repo_root=repo_root).tokenizer_path
        if not tokenizer_src.exists():
            raise FileNotFoundError(f"MS2 tokenizer not found: {tokenizer_src}")
        tokenizer_out = assets / TOKENIZER_FILENAME
        shutil.copyfile(tokenizer_src, tokenizer_out)

        from src.ms3.models.model_b_export import export_model_b_npz

        model_out = export_model_b_npz(checkpoint, assets / MODEL_B_FILENAME)

    return WebExportResult(chunks_path, count, tokenizer_out, model_out)
