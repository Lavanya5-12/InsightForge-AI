from src.ollama_client import default_ollama_client
from src.config import LLM_MODEL


def generate_answer(
    question: str,
    context: str,
    model: str = LLM_MODEL
) -> str:
    """
    Generate a grounded document answer using the local Ollama LLM.
    """
    prompt = f"""
You are InsightForge AI, an intelligent document question-answering assistant.

Answer the user's question using ONLY the provided document context below.

CRITICAL RULES:
1. Do NOT use external or outside knowledge.
2. Do NOT invent facts or hallucinate details.
3. If the answer is not clearly stated in the context below, respond with:
   "I could not find sufficient information about this question in the uploaded documents."
4. Provide a direct, concise, and accurate answer.
5. Do NOT mention "SOURCE 1", "SOURCE 2", or internal retrieval mechanics in your main text.
6. Answer naturally and professionally.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    return default_ollama_client.generate_answer(
        prompt=prompt,
        model=model,
        temperature=0.2
    )