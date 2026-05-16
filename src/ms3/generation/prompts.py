from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# System-Guided Prompts (Strict constraints and behavior definitions)
ARABIC_SYSTEM_GUIDED = """أنت مساعد ذكي ومفيد. مهمتك هي الإجابة على أسئلة المستخدم باستخدام السياق المقدم فقط.
التعليمات:
1. اعتمد فقط على المعلومات الموجودة في السياق أدناه.
2. إذا لم يكن السياق يحتوي على معلومات كافية للإجابة، يجب عليك الرد بـ "OUT_OF_DOMAIN" أو توضيح أن المعلومات غير متوفرة في السياق.
3. يمكنك الإجابة باللغة العربية أو الإنجليزية بناءً على لغة المستخدم.
4. حافظ على دقة المعلومات ولا تقترع أو تهلوس إجابات.

السياق (Context):
{context}
"""

ENGLISH_SYSTEM_GUIDED = """You are a helpful AI assistant. Your task is to answer user questions strictly based on the provided context.
Instructions:
1. Rely ONLY on the information present in the Context.
2. If the context does not contain enough information to answer the question, you MUST reply with exactly "OUT_OF_DOMAIN".
3. Answer in Arabic or English matching the user's language.
4. Do not hallucinate or make up facts.

Context:
{context}
"""

# Minimal Prompts (Basic context grounding)
ARABIC_MINIMAL = """السياق:
{context}

أجب على السؤال بناءً على السياق فقط.
"""

ENGLISH_MINIMAL = """Context:
{context}

Answer the question strictly based on the context. If you don't know, say "OUT_OF_DOMAIN".
"""

def get_prompt(template_type: str = "english_guided"):
    """
    Returns the configured ChatPromptTemplate based on the selected prompt strategy.
    
    Strategies:
    - arabic_guided: Comprehensive Arabic instructions
    - english_guided: Comprehensive English instructions
    - arabic_minimal: Sparse Arabic instructions
    - english_minimal: Sparse English instructions
    """
    templates = {
        "arabic_guided": ARABIC_SYSTEM_GUIDED,
        "english_guided": ENGLISH_SYSTEM_GUIDED,
        "arabic_minimal": ARABIC_MINIMAL,
        "english_minimal": ENGLISH_MINIMAL
    }
    
    if template_type not in templates:
        template_type = "english_guided"
        
    system_prompt = SystemMessagePromptTemplate.from_template(templates[template_type])
    human_prompt = HumanMessagePromptTemplate.from_template("History:\n{history}\n\nQuestion: {question}")
    
    return ChatPromptTemplate.from_messages([system_prompt, human_prompt])
