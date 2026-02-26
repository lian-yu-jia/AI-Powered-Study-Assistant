import requests
from rag import retrieve_context

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini"

def summarize_document(chunks, max_words=150, top_k=3):
    if not chunks:
        return "No content available."

    context = "\n\n".join(chunks[:10])

    prompt = f"""
Summarize the following content clearly and concisely.
Ignore the author and the references.
Focus on the important facts.
Keep the summary under {max_words} words.

Content:
{context}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": max_words + 50}
            },
            timeout=120  # adjust if needed
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        return f"Model error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"
