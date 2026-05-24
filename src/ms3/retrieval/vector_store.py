import os
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class VectorStoreManager:
    """
    Manages the Vector Store (ChromaDB) and the Embedding Model for the MS3 RAG System.
    """
    def __init__(
        self, 
        persist_directory: str = "data/processed/ms3/chroma_db",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Initializes the embedding model and the vector store.
        
        Using a multilingual model to correctly capture Arabic semantics
        while preserving English tokens (code-switching) without lemmatization or stemming
        as per the Milestone 3 constraints.
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        
        # Initialize multilingual embeddings
        # paraphrase-multilingual-MiniLM-L12-v2 is an excellent choice for Arabic+English text
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize Chroma vector store
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="ms1_transcripts"
        )

    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """
        Adds a list of Langchain Documents to the vector store in batches.
        """
        if not documents:
            print("No documents to add.")
            return

        print(f"Adding {len(documents)} chunks to the Vector Store...")
        
        # Process in batches to avoid memory/timeout issues
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.vector_store.add_documents(documents=batch)
            print(f"Processed batch {i // batch_size + 1}...")
            
        print("Documents successfully added to the vector store and persisted.")

    def get_retriever(self, search_type: str = "similarity", k: int = 4):
        """
        Returns a retriever configured to fetch relevant context from the vector store.
        
        Args:
            search_type: Type of search ('similarity', 'mmr', etc.)
            k: Number of documents to return
        """
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

    def get_retriever_for_reranking(self, initial_k: int = 20):
        """
        Returns a retriever that over-fetches documents for subsequent re-ranking.
        """
        return self.get_retriever(search_type="similarity", k=initial_k)
