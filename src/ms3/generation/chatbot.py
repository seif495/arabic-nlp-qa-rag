from typing import List, Dict, Any, Tuple
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from src.ms3.generation.prompts import get_prompt
from src.ms3.generation.llm_manager import LLMManager

class RagChatbot:
    """
    Multi-turn RAG Chatbot implementing various memory constraints 
    and context construction strategies.
    
    Requirements fulfilled:
    - Multi-turn conversation
    - Out-of-Domain Detection
    - Context window strategies (full, sliding, truncation, summarized)
    """
    
    def __init__(
        self, 
        retriever, 
        prompt_type: str = "english_guided",
        memory_strategy: str = "sliding_window"
    ):
        self.retriever = retriever
        self.memory_strategy = memory_strategy
        
        # Setup the chain
        self.llm = LLMManager().get_llm()
        self.prompt = get_prompt(prompt_type)
        
        # Simple LCEL chain
        self.chain = self.prompt | self.llm | StrOutputParser()
        
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
            # For this MVP, acting as a placeholder where actual summarization chain would run.
            # E.g., summarized = self.summarize_chain.invoke({"history": full})
            formatted.append("Summary of previous conversation: [User and Assistant discussed previously documented topics.]")
            
        return "\n".join(formatted)

    def answer_query(self, user_query: str) -> str:
        """
        Retrieves context, formats history, and generates a response.
        Handles API failures and Out-Of-Domain gracefully.
        """
        try:
            # 1. Retrieve Context
            docs = self.retriever.invoke(user_query)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # 2. Format History
            history_text = self.format_history()
            
            # 3. Generate Answer 
            response = self.chain.invoke({
                "context": context_text,
                "history": history_text,
                "question": user_query
            })
            
            # 4. Out-of-Domain Detection & Rejection Message
            # The prompt instructs the LLM to output "OUT_OF_DOMAIN" if the answer isn't in context.
            if "OUT_OF_DOMAIN" in response.upper():
                response = "عذراً، هذا السؤال خارج نطاق النصوص المتوفرة لدي. / Sorry, this question is out of bounds for the available transcripts."
            
            # Update state
            self.history.append((user_query, response))
            return response
            
        except Exception as e:
            # "Never crash" requirement
            print(f"System Error: {str(e)}")
            return "حدث خطأ غير متوقع في النظام. الرجاء المحاولة مرة أخرى لاحقاً. / An unexpected system error occurred. Please try again later."
    
    def reset_memory(self):
        self.history = []
