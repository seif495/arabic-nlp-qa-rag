from typing import List, Dict, Any, Tuple, Optional, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from src.ms3.generation.prompts import get_prompt
from src.ms3.generation.llm_manager import LLMManager
from src.ms2.metrics.normalize import arabic_post_normalize
from src.ms3.retrieval.reranker import MS2Reranker
from src.ms3.models.ms2_bridge import MS2Bridge

class RagChatbot:
    """
    Multi-turn RAG Chatbot implementing various memory constraints 
    and context construction strategies.
    
    Integrated with MS2 models for re-ranking and hybrid generation.
    """
    
    def __init__(
        self, 
        retriever, 
        prompt_type: str = "english_guided",
        memory_strategy: str = "sliding_window",
        ms2_bridge: Optional[MS2Bridge] = None,
        generation_mode: str = "llm_only"
    ):
        self.retriever = retriever
        self.memory_strategy = memory_strategy
        self.ms2_bridge = ms2_bridge
        self.generation_mode = generation_mode
        self.reranker = MS2Reranker(ms2_bridge) if ms2_bridge else None
        
        # Setup the chain
        self.llm = LLMManager().get_llm()
        self.prompt = get_prompt(prompt_type)
        
        # Simple LCEL chain
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        # Standalone Query Chain for Contextualization
        from langchain_core.prompts import PromptTemplate
        self.standalone_query_prompt = PromptTemplate.from_template(
            "Given the following conversation history and the latest user question, "
            "rephrase the question to be a standalone semantic query optimized for finding information in an Egyptian Arabic transcript. "
            "IMPORTANT INSTRUCTIONS:\n"
            "1. Output in Arabic, if the user asked in arabic. Output in english if the user asked in English.\n"
            "2. Fix any spelling mistakes in the user's question.\n"
            "3. Remove unnecessary definite articles (like 'ال' prepended to proper nouns, e.g. 'التاج محل' -> 'تاج محل').\n"
            "4. Use natural Egyptian Arabic vocabulary (e.g. 'اتبني' instead of 'بني').\n"
            "5. Make it a single fluid sentence. No bullet points or boolean operators.\n"
            "If it's just a greeting, leave it as is.\n\n"
            "History:\n{history}\n\n"
            "Latest Question: {question}\n\n"
            "Standalone Query:"
        )
        self.standalone_chain = self.standalone_query_prompt | self.llm | StrOutputParser()
        
        # State: conversation history
        self.history: List[Tuple[str, str]] = []

    def format_history(self) -> str:
        """
        Formats the current conversation history based on the selected memory strategy.
        """
        if not self.history:
            return ""
            
        formatted: List[str] = []
        
        if self.memory_strategy == "full_history":
            # Append everything
            for human, ai in self.history:
                formatted.extend([f"User: {human}", f"Assistant: {ai}"])
                
        elif self.memory_strategy == "sliding_window":
            # Only keep the last 3 turns
            window = self.history[-3:]
            for human, ai in window:
                formatted.extend([f"User: {human}", f"Assistant: {ai}"])
                
        elif self.memory_strategy == "strict_truncation":
            # Limit total characters (e.g., last ~1000 characters)
            raw = ""
            for human, ai in reversed(self.history):
                turn = f"User: {human}\nAssistant: {ai}\n"
                if len(raw) + len(turn) > 1000:
                    break
                raw = turn + raw
            return raw
            
        elif self.memory_strategy == "summarized_history":
            from langchain_core.prompts import PromptTemplate
            
            # Combine full history to summarize
            full_history_text = ""
            for human, ai in self.history:
                full_history_text += f"User: {human}\nAssistant: {ai}\n"
                
            summary_prompt = PromptTemplate.from_template(
                "Summarize the following conversation history briefly. Focus on the main topics discussed. Keep it concise.\n\nConversation History:\n{history}\n\nSummary:"
            )
            summarizer_chain = summary_prompt | self.llm | StrOutputParser()
            
            # Invoke LLM to generate summary on the fly
            summary = summarizer_chain.invoke({"history": full_history_text})
            formatted.append(f"Summary of previous conversation:\n{summary}")
            
        return "\n".join(formatted)

    def answer_query(self, user_query: str, use_reranking: bool = False) -> Union[str, Dict[str, str]]:
        """
        Retrieves context, formats history, and generates a response.
        Handles API failures and Out-Of-Domain gracefully.
        """
        try:
            # 1. Format History for contextualization
            history_text = self.format_history()
            
            # 2. Contextualize query logically (standalone query generation to fix semantics)
            standalone_query = self.standalone_chain.invoke({
                "history": history_text if self.history else "No history. This is the first question.",
                "question": user_query
            })
                
            # 3. Retrieve Context using normalized semantic query
            normalized_query = arabic_post_normalize(standalone_query)
            
            # Retrieval with optional re-ranking
            docs = self.retriever.invoke(normalized_query)
            if use_reranking and self.reranker:
                docs = self.reranker.rerank(normalized_query, docs)
            
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # 4. Generate Answer
            llm_response = None
            ms2_response = None
            
            if self.generation_mode in ("llm_only", "hybrid"):
                llm_response = self.chain.invoke({
                    "context": context_text,
                    "history": history_text,
                    "question": standalone_query
                })
                # Out-of-Domain Detection & Rejection Message
                if "OUT_OF_DOMAIN" in llm_response.upper():
                    llm_response = "عذراً، هذا السؤال خارج نطاق النصوص المتوفرة لدي. / Sorry, this question is out of bounds for the available transcripts."
            
            if self.generation_mode in ("ms2_only", "hybrid"):
                if self.ms2_bridge:
                    ms2_response = self.ms2_bridge.generate_answer(standalone_query, context_text)
                else:
                    ms2_response = "[MS2 model not loaded]"
            
            # Determine final response format
            if self.generation_mode == "hybrid":
                response = {"llm": llm_response, "ms2": ms2_response}
                # Store LLM response in history for consistency
                self.history.append((user_query, llm_response))
                return response
            elif self.generation_mode == "ms2_only":
                response = ms2_response
                self.history.append((user_query, response))
                return response
            else:
                response = llm_response
                self.history.append((user_query, response))
                return response
            
        except Exception as e:
            # "Never crash" requirement
            print(f"System Error: {str(e)}")
            return "حدث خطأ غير متوقع في النظام. الرجاء المحاولة مرة أخرى لاحقاً. / An unexpected system error occurred. Please try again later."
    
    def reset_memory(self):
        self.history = []
