from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

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
