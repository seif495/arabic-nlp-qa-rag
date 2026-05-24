import os
import json
import numpy as np
from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from src.ms3.generation.llm_manager import LLMManager
from src.ms3.retrieval.vector_store import VectorStoreManager

class RagEvaluator:
    """
    Evaluates the RAG system based on three core dimensions designed for MS3:
    1. Text Generation Quality (Fluency & Readability via LLM Judge)
    2. Semantic Correctness (Cosine similarity of embeddings: Generated vs Reference)
    3. Grounding to Retrieved Context (Faithfulness via LLM Judge)
    """
    def __init__(self):
        # We use the embedding model to calculate semantic similarities
        vsm = VectorStoreManager()
        self.embeddings = vsm.embeddings
        
        # We use an LLM for judge-based metrics (Text Quality, Grounding)
        self.llm = LLMManager().get_llm()
        
        self.faithfulness_prompt = PromptTemplate.from_template(
            "You are a strict evaluator. Given the following context and an answer, "
            "determine if the answer is completely grounded in the context. "
            "Answer '1' if fully grounded/faithful, and '0' if it contains hallucinations or ungrounded facts.\n\n"
            "Context: {context}\nAnswer: {answer}\n\nScore (0 or 1):"
        )
        
        self.fluency_prompt = PromptTemplate.from_template(
            "You are a language evaluator. Rate the fluency and grammatical correctness of this Arabic/English text "
            "on a scale from 1 (poor) to 5 (excellent). Just output the number.\n\n"
            "Text: {answer}\n\nScore (1-5):"
        )

    def evaluate_semantic_correctness(self, generated_answer: str, reference_answer: str) -> float:
        """
        Metric 1: Semantic Correctness
        Justification: Exact word-matching (BLEU) fails heavily on Arabic due to dialectal 
        and morphological complexity. Cosine similarity of multilingual embeddings ensures 
        we measure if the *meaning* matches the ground truth MS2 QA pair.
        """
        if not generated_answer or not reference_answer:
            return 0.0
            
        emb_gen = self.embeddings.embed_query(generated_answer)
        emb_ref = self.embeddings.embed_query(reference_answer)
        
        # Cosine similarity
        dot_product = np.dot(emb_gen, emb_ref)
        norm_gen = np.linalg.norm(emb_gen)
        norm_ref = np.linalg.norm(emb_ref)
        
        if norm_gen == 0 or norm_ref == 0:
            return 0.0
            
        return float(dot_product / (norm_gen * norm_ref))

    def evaluate_grounding(self, generated_answer: str, context: str) -> int:
        """
        Metric 2: Grounding / Faithfulness 
        Justification: In RAG, hallucination is a critical failure. Using an LLM-as-a-judge 
        mimics RAGAS frameworks to strictly verify if the answer statements span from context.
        """
        if "OUT_OF_DOMAIN" in generated_answer.upper() or "خارج نطاق" in generated_answer:
            return 1 # Successfully rejected OOD, which is perfectly grounded behavior
            
        chain = self.faithfulness_prompt | self.llm
        try:
            result = chain.invoke({"context": context, "answer": generated_answer})
            return 1 if "1" in str(result.content) else 0
        except:
            return 0

    def evaluate_generation_quality(self, generated_answer: str) -> int:
        """
        Metric 3: Text Generation Quality
        Justification: Preserving English code-switching and dialectal variations can map 
        badly to standard readability formulas. Passing to an LLM evaluator helps grade fluency.
        """
        chain = self.fluency_prompt | self.llm
        try:
            result = chain.invoke({"answer": generated_answer})
            score_str = str(result.content).strip()
            # extract number
            for s in ["5", "4", "3", "2", "1"]:
                if s in score_str: return int(s)
            return 1
        except:
            return 1
    
    def run_evaluation_suite(self, qa_pairs: List[Dict[str, str]], chatbot) -> Dict[str, Any]:
        """
        Runs the chatbot over MS2 reference QA pairs and aggregates the scores.
        qa_pairs format: [{"question": "...", "answer": "..."}]
        """
        results = {
            "semantic_correctness": [],
            "grounding": [],
            "generation_quality": []
        }
        
        for idx, item in enumerate(qa_pairs):
            q = item["question"]
            ref_a = item["answer"]
            
            # Reset chatbot memory per independent evaluation
            chatbot.reset_memory()
            
            # Get Context manually to pass to evaluator
            from src.ms2.metrics.normalize import arabic_post_normalize
            docs = chatbot.retriever.invoke(arabic_post_normalize(q))
            context = "\n".join([doc.page_content for doc in docs])
            
            # Generate
            gen_a = chatbot.answer_query(q)
            
            # Score
            sem = self.evaluate_semantic_correctness(gen_a, ref_a)
            grnd = self.evaluate_grounding(gen_a, context)
            qual = self.evaluate_generation_quality(gen_a)
            
            results["semantic_correctness"].append(sem)
            results["grounding"].append(grnd)
            results["generation_quality"].append(qual)
            
            print(f"Eval {idx+1}/{len(qa_pairs)} | Sem: {sem:.2f} | Grnd: {grnd} | Qual: {qual}")
            
        # Aggregate
        summary = {
            "avg_semantic_correctness": float(np.mean(results["semantic_correctness"])),
            "grounding_success_rate": float(np.mean(results["grounding"])),
            "avg_generation_quality_out_of_5": float(np.mean(results["generation_quality"]))
        }
        return summary
