import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def generate_answer(
    question: str,
    context: str
) -> str:

    prompt = f"""
You are InsightForge AI, a document question-answering assistant.

Answer the user's question using ONLY the information provided
in the context below.

IMPORTANT RULES:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not present in the context, say:
   "I could not find the answer in the provided document."
4. Give a direct, clear answer.
5. Do not mention "SOURCE 1", "SOURCE 2", or internal retrieval
   details in your answer.
6. Do not mention the context or retrieval process.
7. Answer naturally as if you already know the information
   from the provided document.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["response"].strip()