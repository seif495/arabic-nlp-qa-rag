import argparse
import os
import glob
from src.ms3.retrieval.document_loader import DocumentLoader
from src.ms3.retrieval.vector_store import VectorStoreManager

def init_vectorstore(args):
    """
    Initializes the Vector Store by loading MS1 transcripts, chunking them,
    and persisting the embeddings.
    """
    input_dir = args.input_dir
    persist_dir = args.persist_dir
    num_episodes = args.num_episodes
    
    # Locate transcripts (Assuming JSON structured output from MS1 but fallback to txt exists in loader)
    all_files = glob.glob(os.path.join(input_dir, "*.json"))
    if not all_files:
        all_files = glob.glob(os.path.join(input_dir, "*.txt"))
        
    if not all_files:
        print(f"No transcript files found in {input_dir}")
        return

    # Select exactly the required number of episodes
    selected_files = all_files[:num_episodes]
    print(f"Selected {len(selected_files)} episodes for the vector store.")
    
    # Process and Chunk Documents
    loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)
    chunks = loader.load_and_chunk_episodes(selected_files)
    
    # Embed and Persist
    vsm = VectorStoreManager(persist_directory=persist_dir)
    vsm.add_documents(chunks, batch_size=100)

def export_web(args):
    """
    Writes the Vercel deployment assets (chunks, MS2 tokenizer, ranked MS2 model weights).
    """
    from pathlib import Path
    from src.ms3.web_export import export_web_assets

    result = export_web_assets(
        repo_root=Path.cwd(),
        export_models=not args.skip_models,
        normalized_dir=Path(args.input_dir),
    )
    print(f"[export-web] chunks={result.chunk_count} path={result.chunks_path}")
    print(f"[export-web] tokenizer={result.tokenizer_path}")
    for name, path in result.model_paths.items():
        print(f"[export-web] model={name} path={path}")
    print(f"[export-web] default_model={result.default_model}")

def train_ms2_web(args):
    """
    Trains MS2 candidate model(s) from the MS1 dataset for the Vercel site (needs TensorFlow).
    """
    from pathlib import Path
    from src.ms3.models.ms2_web_selection import CANDIDATES, train_candidate

    names = CANDIDATES if args.model == "all" else (args.model,)
    for name in names:
        best = train_candidate(Path.cwd(), name, budget_minutes=args.budget_minutes, batch_size=args.batch_size, seed=args.seed)
        print(f"[train-ms2-web] model={name} best_checkpoint={best}")

def select_ms2_web(args):
    """
    Scores every trained MS2 candidate on the dev split and picks the website model.
    """
    from pathlib import Path
    from src.ms3.models.ms2_web_selection import SELECTION_REPORT, select_model

    ranked = select_model(Path.cwd(), seed=args.seed)
    for score in ranked:
        print(f"[select-ms2-web] model={score.model} dev_token_f1={score.dev_token_f1} dev_em={score.dev_em} dev_loss={score.best_dev_loss}")
    print(f"[select-ms2-web] selected={ranked[0].model} report={SELECTION_REPORT}")

def run_app(args):
    """
    Boots up the Streamlit UI.
    """
    import subprocess
    print("Starting Streamlit App...")
    subprocess.run(["streamlit", "run", "src/ms3/app/app.py"])

def main():
    parser = argparse.ArgumentParser(description="MS3 RAG System CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Vector store initialization parser
    init_parser = subparsers.add_parser("init-vectorstore", help="Initialize the ChromaDB by processing MS1 transcripts")
    init_parser.add_argument("--input_dir", type=str, default="data/interim/normalized", help="Directory containing MS1 normalized transcripts (output of `ms1 normalize`)")
    init_parser.add_argument("--persist_dir", type=str, default="data/processed/ms3/chroma_db", help="Directory to persist the Chroma Vector Store")
    init_parser.add_argument("--num_episodes", type=int, default=13, help="Number of episodes to use (constraint: up to 13)")

    # MS2 model selection for the website
    train_parser = subparsers.add_parser("train-ms2-web", help="Train MS2 candidate model(s) from the MS1 dataset (needs TensorFlow)")
    train_parser.add_argument("--model", choices=["all", "model_a", "rnn_model", "model_b", "transformer"], default="all")
    train_parser.add_argument("--budget_minutes", type=float, default=35.0, help="Per-model upper bound; early stopping on dev loss usually ends sooner")
    train_parser.add_argument("--batch_size", type=int, default=16)
    train_parser.add_argument("--seed", type=int, default=42)
    select_parser = subparsers.add_parser("select-ms2-web", help="Score trained MS2 candidates on dev and write docs/reports/ms3_web_model_selection.md")
    select_parser.add_argument("--seed", type=int, default=42)

    # Vercel web assets parser
    export_parser = subparsers.add_parser("export-web", help="Export chunks + the ranked MS2 models' weights for the Vercel site in web/")
    export_parser.add_argument("--input_dir", type=str, default="data/interim/normalized", help="Directory containing MS1 normalized transcripts")
    export_parser.add_argument("--skip_models", action="store_true", help="Only export the transcript chunks")

    # Streamlit parser
    run_parser = subparsers.add_parser("run-app", help="Run the Streamlit Web UI")
    
    args = parser.parse_args()
    
    if args.command == "init-vectorstore":
        init_vectorstore(args)
    elif args.command == "train-ms2-web":
        train_ms2_web(args)
    elif args.command == "select-ms2-web":
        select_ms2_web(args)
    elif args.command == "export-web":
        export_web(args)
    elif args.command == "run-app":
        run_app(args)

if __name__ == "__main__":
    main()
