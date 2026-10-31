"""Build the static assets the Vercel deployment (``web/``) ships with.

``python -m src.cli.ms3 export-web`` writes into ``web/ms3web/assets/``:

- ``ms3_chunks.json``: MS1-normalised transcripts chunked exactly like the
  MS3 vector store (``DocumentLoader(chunk_size=1000, chunk_overlap=200)``);
- ``ms2_tokenizer.model`` / ``ms2_char_vocab.json``: the MS2 SentencePiece
  model and character vocabulary the MS2 models were trained with;
- ``ms2_<model>.npz``: trained weights of every MS2 model with a NumPy web
  runtime (``model_a``, ``model_b``, ``transformer``);
- ``ms2_models.json``: those models ranked by ``select-ms2-web`` (dev
  token-F1) with their scores; the winner is the site's default.
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
CHAR_VOCAB_FILENAME = "ms2_char_vocab.json"
MODELS_INDEX_FILENAME = "ms2_models.json"
DISPLAY_NAMES = {
    "model_a": "Model A (RNN + FiLM)",
    "model_b": "Model B (RoPE Transformer)",
    "transformer": "Vanilla Transformer (sinusoidal)",
}
POSITIONAL_ENCODING = {"model_a": "recurrent", "model_b": "rope", "transformer": "sinusoidal"}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class WebExportResult:
    chunks_path: Path
    chunk_count: int
    tokenizer_path: Path | None
    model_paths: dict[str, Path]
    default_model: str | None


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
    export_models: bool = True,
    normalized_dir: Path = Path("data/interim/normalized"),
    seed: int = 42,
) -> WebExportResult:
    repo_root = repo_root.resolve()
    assets = repo_root / WEB_ASSETS_RELATIVE
    chunks_path = assets / CHUNKS_FILENAME
    count = export_chunks(repo_root / normalized_dir, chunks_path)
    if not export_models:
        return WebExportResult(chunks_path, count, None, {}, None)

    from src.ms3.models.ms2_web_export import EXPORTABLE, RUNTIMES, export_npz
    from src.ms3.models.ms2_web_selection import SELECTION_JSON, checkpoint_path

    selection_path = repo_root / SELECTION_JSON
    if not selection_path.exists():
        raise FileNotFoundError(
            f"{selection_path} missing; run `python -m src.cli.ms3 train-ms2-web` then `select-ms2-web`"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    ranked = [c for c in selection["candidates"] if c["model"] in EXPORTABLE]
    if not ranked:
        raise ValueError("no exportable MS2 model in the selection results")

    paths = resolve_ms2_paths(repo_root=repo_root)
    tokenizer_out = assets / TOKENIZER_FILENAME
    shutil.copyfile(paths.tokenizer_path, tokenizer_out)
    shutil.copyfile(paths.char_vocab_path, assets / CHAR_VOCAB_FILENAME)

    model_paths: dict[str, Path] = {}
    index = []
    for rank, candidate in enumerate(ranked, start=1):
        name = candidate["model"]
        out = assets / f"ms2_{name}.npz"
        model_paths[name] = export_npz(name, checkpoint_path(repo_root, name, seed), out)
        index.append(
            {
                "model": name,
                "display_name": DISPLAY_NAMES[name],
                "description": candidate["description"],
                "positional_encoding": POSITIONAL_ENCODING[name],
                "runtime": RUNTIMES[name],
                "weights": out.name,
                "rank": rank,
                "parameter_count": candidate["parameter_count"],
                "dev_em": candidate["dev_em"],
                "dev_token_f1": candidate["dev_token_f1"],
            }
        )
    default = index[0]["model"]
    (assets / MODELS_INDEX_FILENAME).write_text(
        json.dumps({"default": default, "selection_metric": selection["metric"], "models": index}, indent=2) + "\n",
        encoding="utf-8",
    )
    return WebExportResult(chunks_path, count, tokenizer_out, model_paths, default)
