import os
import json
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentLoader:
    """
    Handles loading MS1 transcripts and applying the chunking strategy 
    for the Retrieval Augmented Generation system.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initializes the document loader and text splitter.
        
        Chunking Strategy Justification:
        - RecursiveCharacterTextSplitter is used to preserve semantic coherence by 
          splitting on logical boundaries (paragraphs, then sentences, then words).
        - chunk_size=1000 and chunk_overlap=200 provide a good balance between 
          capturing enough context for the LLM to understand semantic meaning, 
          while keeping chunks small enough to fit within context windows and 
          yield precise vector matches.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Using default separators which prioritize \n\n, \n, spaces
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "؟", "!", " ", ""]
        )

    def load_and_chunk_episodes(self, episode_paths: List[str]) -> List[Document]:
        """
        Loads the provided transcripts and splits them into traceable chunks.
        
        Args:
            episode_paths: Paths to 3-5 MS1 normalized transcript files.
            
        Returns:
            A list of Langchain Documents with content and traceability metadata.
        """
        all_chunks = []
        
        for file_path in episode_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}")
                continue
                
            episode_name = os.path.basename(file_path)
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # If the format is JSON (e.g., list of dictionary lines), we parse it.
            # If it's a raw text file, we just read the text.
            try:
                # Attempt to parse as JSON if transcripts are structured
                data = json.loads(content)
                if isinstance(data, list):
                    # Combine structured utterances into coherent text blocks
                    # assuming each item has something like 'text', 'speaker', etc.
                    # Adjust dictionary keys based on MS1 schema
                    text_blocks = []
                    for item in data:
                        text = item.get("normalized_text", item.get("text", ""))
                        if text:
                            text_blocks.append(text)
                    raw_text = "\n".join(text_blocks)
                else:
                    raw_text = content
            except json.JSONDecodeError:
                # Fallback to pure text
                raw_text = content
                
            # Create a base document with the metadata tracking where it came from
            source_doc = Document(
                page_content=raw_text,
                metadata={"source_episode": episode_name}
            )
            
            # Split the document
            chunks = self.text_splitter.split_documents([source_doc])
            
            # Enrich metadata with chunk IDs to ensure exact traceability
            for idx, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = idx
                all_chunks.append(chunk)
                
        print(f"Loaded {len(episode_paths)} episodes into {len(all_chunks)} semantic chunks.")
        return all_chunks
