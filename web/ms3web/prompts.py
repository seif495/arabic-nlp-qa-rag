"""Prompt templates for the web chatbot.

The four answer-generation templates are verbatim copies of
``src/ms3/generation/prompts.py`` (kept in sync by ``tests/web/test_web_sync.py``)
so the deployed site evaluates the same MS3 prompt-engineering strategies.
"""

from __future__ import annotations

# System-Guided Prompts (Strict constraints and behavior definitions)
ARABIC_SYSTEM_GUIDED = """أنت مساعد ذكي ومفيد. مهمتك هي الإجابة على أسئلة المستخدم باستخدام السياق المقدم فقط.
التعليمات:
1. اعتمد فقط على المعلومات الموجودة في السياق أدناه.
2. إذا لم يكن السياق يحتوي على معلومات كافية للإجابة على سؤال واقعي، يجب عليك الرد بـ "OUT_OF_DOMAIN".
3. إذا كان المستخدم يقوم بالترحيب ("مرحبا"، "كيف حالك"، "hey")، قم بالرد بترحيب مهذب ولا تستخدم "OUT_OF_DOMAIN".
4. يمكنك الإجابة باللغة العربية أو الإنجليزية بناءً على لغة المستخدم.
5. حافظ على دقة المعلومات ولا تقترع أو تهلوس إجابات.

السياق (Context):
{context}
"""

ENGLISH_SYSTEM_GUIDED = """You are a helpful AI assistant. Your task is to answer user questions strictly based on the provided context.
Instructions:
1. Rely ONLY on the information present in the Context.
2. If the context does not contain enough information to answer a factual question, you MUST reply with exactly "OUT_OF_DOMAIN".
3. If the user is just greeting you or chatting (e.g. "hey", "hello", "how are you"), respond with a polite greeting and do NOT output "OUT_OF_DOMAIN".
4. Answer in Arabic or English matching the user's language.
5. Do not hallucinate or make up facts.

Context:
{context}
"""

# Minimal Prompts (Basic context grounding)
ARABIC_MINIMAL = """السياق:
{context}

أجب على السؤال بناءً على السياق فقط. إذا كان مجرد ترحيب، رد بالترحيب. 
"""

ENGLISH_MINIMAL = """Context:
{context}

Answer the question strictly based on the context. If it's a greeting, reply politely. If it's a factual question and you don't know, say "OUT_OF_DOMAIN".
"""

TEMPLATES = {
    "arabic_guided": ARABIC_SYSTEM_GUIDED,
    "english_guided": ENGLISH_SYSTEM_GUIDED,
    "arabic_minimal": ARABIC_MINIMAL,
    "english_minimal": ENGLISH_MINIMAL,
}

HUMAN_TEMPLATE = "History:\n{history}\n\nQuestion: {question}"

# Adapted from the MS3 standalone-query chain (src/ms3/generation/chatbot.py).
# The web retriever is lexical (BM25 over Arabic transcripts), so the
# retrieval query is always rewritten into Arabic, whatever language the user
# typed; the answer still follows the user's language.
STANDALONE_QUERY_TEMPLATE = (
    "Given the following conversation history and the latest user question, "
    "rephrase the question to be a standalone semantic query optimized for finding "
    "information in an Egyptian Arabic transcript. "
    "IMPORTANT INSTRUCTIONS:\n"
    "1. ALWAYS output in Arabic, even if the user asked in English (translate it). "
    "Keep names that are usually written in English (e.g. Citizen Kane) as they are.\n"
    "2. Fix any spelling mistakes in the user's question.\n"
    "3. Remove unnecessary definite articles (like 'ال' prepended to proper nouns, "
    "e.g. 'التاج محل' -> 'تاج محل').\n"
    "4. Use natural Egyptian Arabic vocabulary (e.g. 'اتبني' instead of 'بني').\n"
    "5. Make it a single fluid sentence. No bullet points or boolean operators.\n"
    "6. Output ONLY the query, with no explanation.\n"
    "If it's just a greeting, leave it as is.\n\n"
    "History:\n{history}\n\n"
    "Latest Question: {question}\n\n"
    "Standalone Query:"
)

SUMMARY_TEMPLATE = (
    "Summarize the following conversation history briefly. Focus on the main topics "
    "discussed. Keep it concise.\n\nConversation History:\n{history}\n\nSummary:"
)


def fill(template: str, **values: str) -> str:
    """``str.format`` without choking on braces inside user text or context."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template
