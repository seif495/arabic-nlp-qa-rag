# Milestone 3: Retrieval-Augmented Generation (RAG) System - Technical Report

## 1. Executive Summary
This report details the design, implementation, and evaluation of the Retrieval-Augmented Generation (RAG) system for Milestone 3. The objective was to build a robust, multi-turn QA pipeline over dialectal Arabic transcripts (El-Da7ee7 episodes) strictly adhering to the text-representation constraints established in Milestone 1. The system successfully implements advanced context-window strategies, robust out-of-domain detection, and automated LLM provider fallbacks (Groq → Gemini). Additionally, extensive testing revealed critical insights into the limitations of Dense Retrieval when applying Modern Standard Arabic (MSA) queries against heavily dialectal, conversational texts.

## 2. System Architecture & Setup

### 2.1 Vector Store and Embeddings
The foundation of the RAG system is built upon **ChromaDB**, operating entirely locally. For the embedding layer, the system utilizes the `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model from HuggingFace. This lightweight, multilingual model was chosen to map both English and Arabic queries into a shared semantic space without requiring external API calls for embedding generation. 

Five normalized transcripts (e.g., *The Octopus*, *The Samurai*, *Taj Mahal*, *John F. Kennedy*) were chunked using a `RecursiveCharacterTextSplitter` with a `chunk_size` of 1000 characters and a `chunk_overlap` of 200 characters.

### 2.2 LLM Orchestration & Fallback Mechanisms
To ensure high availability and prevent application crashes due to API rate limits, the system implements an automated fallback architecture using LangChain:
*   **Primary LLM:** `ChatGroq` utilizing the `llama-3.3-70b-versatile` model. Chosen for its extreme inference speed and strong multilingual reasoning capabilities.
*   **Fallback LLM:** `ChatGoogleGenerativeAI` utilizing the `gemini-1.5-flash` model. 
Using LangChain's `.with_fallbacks()` method, the system seamlessly routes requests to the Gemini API if the Groq API encounters rate-limiting or connection failures, satisfying the "Never crash" requirement of the rubric.

---

## 3. Data Representation & MS1 Constraints

A core challenge of this project was preserving the linguistic integrity of the original transcripts. The MS1 normalization pipeline was executed successfully prior to vector insertion, strictly adhering to the following rules:
1.  **No Lemmatization or Stemming:** To preserve the Egyptian colloquial structure.
2.  **No Punctuation Removal:** To maintain conversational flow and sentence boundaries, which proved critical for the Recursive Character Splitter.
3.  **No Removal of English Tokens:** Preserving the Arabizi and code-switching frequent in modern Arabic media.

By retaining these features, the vector store accurately reflects the dialectal nuances of the source material. However, this strict adherence introduced unique retrieval challenges (discussed in Section 5).

---

## 4. Prompt Engineering & Out-of-Domain Guardrails

The system employs strict prompt engineering to mitigate LLM hallucinations and enforce grounding in the retrieved context. 

### 4.1 Strict Grounding
The system prompt explicitly commands the LLM: 
> *"If the context does not contain enough information to answer the question, you MUST reply with exactly 'OUT_OF_DOMAIN'."*

During testing, when the system was asked meta-questions (e.g., *"How many transcripts are there?"* or *"List all episode titles"*), the LLM correctly identified that the retrieved k=4 chunks lacked this macroscopic information. Consequently, it output `OUT_OF_DOMAIN`, which the application intercepted to display the designed rejection message: 
*"عذراً، هذا السؤال خارج نطاق النصوص المتوفرة لدي."*

### 4.2 Hallucination Control via Prompt Variations
An A/B test was conducted comparing an expansive prompt (`english_guided`) against a restrictive prompt (`arabic_minimal`). When asked *"Why do extremists get angry about the Taj Mahal?"*:
*   The `english_guided` prompt hallucinated slightly, claiming the Taj Mahal was a symbol of "Hinduism."
*   The `arabic_minimal` prompt remained perfectly grounded, stating it was a "global Indian symbol," exactly mirroring the text. This demonstrates the direct impact of prompt verbosity on the hallucination rate of 70B parameter models.

---

## 5. Context Window & Memory Strategies

To support multi-turn conversations without exhausting the LLM token window, three memory strategies were implemented and validated:

1.  **Sliding Window (`sliding_window`):** Retains only the last 3 `(Human, AI)` interaction turns. This is highly efficient for short follow-up questions but suffers from context amnesia in longer sessions.
2.  **Strict Truncation (`strict_truncation`):** Caps the history strictly by character limit (e.g., 1000 characters).
3.  **Summarized History (`summarized_history`):** The most sophisticated strategy. Before querying the main RAG chain, the system dynamically invokes the LLM to generate a concise summary of the entire conversation history. This summary is injected into the prompt. 

**Validation:** The summarized history was proven highly effective. When the summary *"The anger of extremists towards the Taj Mahal was discussed..."* was passed to the LLM alongside a vague query (*"w ba3deen?"*), the LLM successfully maintained the topic and retrieved deep contextual facts (the destruction of the Babri Mosque) to continue the conversation.

---

## 6. Evaluation: Limitations of Dense Retrieval on Dialectal Text

The most significant technical finding of Milestone 3 was the fragility of Dense Retrieval when querying colloquial Arabic (Egyptian dialect) using Formal/Modern Standard Arabic (MSA) or English.

Because the system strictly preserved dialectal variations (MS1 constraint), factual answers were often buried within conversational storytelling. For example, the fact that Shah Jahan built the Taj Mahal was encoded colloquially as *"هيبنيلها اعظم صرح"* (he will build for her the greatest edifice), heavily surrounded by jokes about Romeo and Juliet.

When the system was queried with the formal MSA question: *"من الذي بنى تاج محل ولماذا؟"* (Who built the Taj Mahal and why?):
1.  The 117M parameter `MiniLM` model failed to establish a strong semantic link between the formal verb *"بنى"* (built) and the colloquial *"هيبنيلها"*.
2.  The dense chunks containing the factual answer were ranked lower than noisy chunks from other episodes that simply repeated the keyword "تاج محل".
3.  As a result, the correct chunk was not passed to the LLM, triggering an `OUT_OF_DOMAIN` rejection.

**Conclusion:** Small-parameter multilingual dense embedding models struggle with cross-dialect semantic mapping (MSA to Egyptian Colloquial) in conversational text. To improve factual retrieval in such datasets, a hybrid retrieval strategy combining BM25 (keyword matching) with Dense Embeddings, or utilizing a larger Arabic-specific embedding model (e.g., MARBERT), is highly recommended for future iterations.
