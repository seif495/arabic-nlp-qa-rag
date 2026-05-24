from typing import List
from langchain_core.documents import Document
from src.ms3.models.ms2_bridge import MS2Bridge

class MS2Reranker:
    """
    Reranks documents using MS2 model scores.
    """
    def __init__(self, bridge: MS2Bridge):
        self.bridge = bridge

    def rerank(self, query: str, documents: List[Document], top_k: int = 4) -> List[Document]:
        """
        Takes a query and a list of documents, returns top_k documents after re-ranking.
        """
        if not documents:
            return []
        
        chunks = [doc.page_content for doc in documents]
        scores = self.bridge.encode_for_reranking(query, chunks)
        
        # Sort documents by score descending
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in doc_scores[:top_k]]
