import os
import sys
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Ensure the repository root is on sys.path so `src` package is importable
# when running `streamlit` from the `src/ms3/app` directory.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ms3.retrieval.vector_store import VectorStoreManager
from src.ms3.generation.chatbot import RagChatbot
from src.ms3.models.ms2_bridge import MS2Bridge
from src.ms3.models.train_ms2 import train_quick_ms2

# App Configuration
st.set_page_config(page_title="Arabic NLP RAG (MS3)", layout="wide")

@st.cache_resource
def get_retriever():
    vsm = VectorStoreManager(persist_directory="data/processed/ms3/chroma_db")
    # Restore stable k=30
    return vsm.get_retriever(k=30)

def init_chatbot(prompt_type: str, memory_strategy: str, ms2_model_type: str, gen_mode: str):
    retriever = get_retriever()
    ms2_bridge = None
    if ms2_model_type != "None":
        ms2_bridge = MS2Bridge(model_type=ms2_model_type.lower().replace(" ", "_"))
    
    # Map display names to internal names
    mode_map = {
        "LLM Only": "llm_only",
        "MS2 Only": "ms2_only",
        "Hybrid": "hybrid"
    }
    
    return RagChatbot(
        retriever=retriever, 
        prompt_type=prompt_type,
        memory_strategy=memory_strategy,
        ms2_bridge=ms2_bridge,
        generation_mode=mode_map.get(gen_mode, "llm_only")
    )

def main():
    st.title("Arabic Code-Switched RAG Chatbot")
    st.markdown("MS3 Submission • Evaluates strictly on retrieved MS1 contexts.")
    
    # ---------------- Sidebar Configuration ----------------
    with st.sidebar:
        st.header("System Settings")
        
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

        st.divider()
        st.header("MS2 Integration")
        
        ms2_model = st.selectbox(
            "MS2 Model Architecture",
            options=["None", "Model A", "Model B", "Transformer"],
            index=0,
            help="Select an MS2 model to use for re-ranking or generation."
        )
        
        gen_mode = st.radio(
            "Generation Mode",
            options=["LLM Only", "MS2 Only", "Hybrid"],
            index=0,
            disabled=(ms2_model == "None")
        )
        
        use_reranking = st.checkbox(
            "Enable MS2 Re-ranking",
            value=False,
            disabled=(ms2_model == "None"),
            help="Use the selected MS2 model to re-rank chunks from ChromaDB."
        )

        st.divider()
        st.subheader("Training")
        train_budget = st.slider("Budget (minutes)", 0.5, 5.0, 1.0, 0.5)
        if st.button("Train Selected MS2 Model"):
            if ms2_model == "None":
                st.error("Please select a model architecture to train.")
            else:
                with st.status(f"Training {ms2_model}...", expanded=True) as status:
                    ckpt = train_quick_ms2(
                        model_type=ms2_model.lower().replace(" ", "_"),
                        budget_minutes=train_budget
                    )
                    status.update(label=f"Training complete! Checkpoint: {ckpt}", state="complete", expanded=False)
                    st.success(f"Model {ms2_model} trained and saved.")
                    # Re-init chatbot to load new weights
                    st.session_state.chatbot = init_chatbot(prompt_strat, mem_strat, ms2_model, gen_mode)

        st.divider()
        if st.button("Reset Conversation"):
            st.session_state.messages = [
                {"role": "assistant", "content": "مرحباً! أنا المساعد الذكي الخاص بك للإجابة على أسئلتك. كيف يمكنني مساعدتك اليوم؟ / Hello! I am your AI assistant. How can I help you today?"}
            ]
            st.session_state.chatbot = init_chatbot(prompt_strat, mem_strat, ms2_model, gen_mode)
            st.rerun()

    # ---------------- Initialize State ----------------
    if "messages" not in st.session_state or not st.session_state.messages:
        st.session_state.messages = [
            {"role": "assistant", "content": "مرحباً! أنا المساعد الذكي الخاص بك للإجابة على أسئلتك. كيف يمكنني مساعدتك اليوم؟ / Hello! I am your AI assistant. How can I help you today?"}
        ]
        
    # Check for state changes that require chatbot re-initialization
    current_params = (prompt_strat, mem_strat, ms2_model, gen_mode)
    if "chatbot" not in st.session_state or st.session_state.get("prev_params") != current_params:
        st.session_state.chatbot = init_chatbot(prompt_strat, mem_strat, ms2_model, gen_mode)
        st.session_state.prev_params = current_params

    # ---------------- Chat Interface ----------------
    # Display chat messages from history on app rerun
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if isinstance(content, dict):
                col1, col2 = st.columns(2)
                with col1:
                    st.info("**LLM Response**")
                    st.markdown(content.get("llm", ""))
                with col2:
                    st.success("**MS2 Response**")
                    st.markdown(content.get("ms2", ""))
            else:
                st.markdown(content)

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
                response = chatbot.answer_query(user_query, use_reranking=use_reranking)
                
                if isinstance(response, dict):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info("**LLM Response**")
                        st.markdown(response.get("llm", ""))
                    with col2:
                        st.success("**MS2 Response**")
                        st.markdown(response.get("ms2", ""))
                else:
                    st.markdown(response)
                
                # Render metadata/logs (Satisfies requirement "Displays outputs and logs")
                with st.expander("🛠️ Show Retrieval & Context Logs"):
                    st.text("Last turn history passed to LLM:")
                    st.code(chatbot.format_history() or "(no history yet)", language=None)
                    st.text(f"Memory strategy: {chatbot.memory_strategy}")
                    st.text(f"Generation mode: {chatbot.generation_mode}")
                    st.text(f"Re-ranking enabled: {use_reranking}")
                    st.text(f"History turns stored: {len(chatbot.history)}")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
