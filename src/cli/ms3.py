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
    # The requirement says "Use only the original transcripts from MS1. You may use 3-5 episodes only."
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
    init_parser.add_argument("--input_dir", type=str, default="data/processed/ms1", help="Directory containing MS1 processed transcripts")
    init_parser.add_argument("--persist_dir", type=str, default="data/processed/ms3/chroma_db", help="Directory to persist the Chroma Vector Store")
    init_parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes to use (constraint: 3-5)")

    # Streamlit parser
    run_parser = subparsers.add_parser("run-app", help="Run the Streamlit Web UI")
    
    args = parser.parse_args()
    
    if args.command == "init-vectorstore":
        init_vectorstore(args)
    elif args.command == "run-app":
        run_app(args)

if __name__ == "__main__":
    main()
