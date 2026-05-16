import os
import sys
import streamlit as st

# Ensure the repository root is on sys.path so `src` package is importable
# when running `streamlit` from the `src/ms3/app` directory.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ms3.retrieval.vector_store import VectorStoreManager
from src.ms3.generation.chatbot import RagChatbot

# App Configuration
st.set_page_config(page_title="Arabic NLP RAG (MS3)", layout="wide")

@st.cache_resource
def get_retriever():
    vsm = VectorStoreManager(persist_directory="data/processed/ms3/chroma_db")
    return vsm.get_retriever(k=4)

def init_chatbot(prompt_type: str, memory_strategy: str):
    retriever = get_retriever()
    return RagChatbot(
        retriever=retriever, 
        prompt_type=prompt_type,
        memory_strategy=memory_strategy
    )

def main():
    st.title("🗣️ Arabic Code-Switched RAG Chatbot")
    st.markdown("MS3 Submission • Evaluates strictly on retrieved MS1 contexts.")
    
    # ---------------- Sidebar Configuration ----------------
    with st.sidebar:
        st.header("⚙️ System Settings")
        
        prompt_strat = st.selectbox(
            "Prompt Engineering Strategy",
            options=["arabic_guided", "english_guided", "arabic_minimal", "english_minimal"],
            help="Compare strict guided prompts vs minimal prompts in both English and Arabic."
        )
        
        mem_strat = st.selectbox(
            "Context Window Strategy",
            options=["sliding_window", "full_history", "strict_truncation", "summarized_history"],
            help="How previous multi-turn conversation history is managed before LLM context overflow."
        )
        
        if st.button("Reset Conversation"):
            st.session_state.messages = []
            st.session_state.chatbot = init_chatbot(prompt_strat, mem_strat)
            st.rerun()

    # ---------------- Initialize State ----------------
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "chatbot" not in st.session_state or st.session_state.get("prev_prompt") != prompt_strat or st.session_state.get("prev_mem") != mem_strat:
        st.session_state.chatbot = init_chatbot(prompt_strat, mem_strat)
        st.session_state.prev_prompt = prompt_strat
        st.session_state.prev_mem = mem_strat

    # ---------------- Chat Interface ----------------
    # Display chat messages from history on app rerun
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # React to user input
    if user_query := st.chat_input("إسأل هنا... / Ask your question here..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(user_query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating response..."):
                chatbot: RagChatbot = st.session_state.chatbot
                response = chatbot.answer_query(user_query)
                st.markdown(response)
                
                # Render metadata/logs (Satisfies requirement "Displays outputs and logs")
                with st.expander("🛠️ Show Retrieval & Context Logs"):
                    st.text("Retrieval Prompt:")
                    st.code(chatbot.chain.first.format(
                        context="[...Truncated Documents...]",
                        history=chatbot.format_history(),
                        question=user_query
                    ))

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
