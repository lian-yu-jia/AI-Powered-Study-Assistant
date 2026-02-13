import requests
from sentence_transformers import SentenceTransformer
from vectorstore import VectorStore

# Ollama config
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini"

# Load embedding model once
import streamlit as st

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()




def retrieve_context(question, top_k=5):
    vectorstore = VectorStore.load("vectorstore.pkl")
    query_embedding = embed_model.encode(question)
    return vectorstore.query(query_embedding, top_k=top_k)


def generate_answer(question, context_chunks):
    if not context_chunks:
        return "I don't know"

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful study assistant.
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code != 200:
        return "Error: Unable to reach language model."

    try:
        data = response.json()
        return data.get("response", "No response generated.")
    except Exception:
        return "Error: Invalid response from language model."


def generate_summary(chunks):
    if not chunks:
        return "No content available to summarize."

    joined_chunks = "\n\n".join(chunks)

    prompt = f"""
Summarize the following study material in 5–7 bullet points.
Be concise.


{joined_chunks}
"""

    response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 200,
            "temperature": 0.3
        }
    }
)

    if response.status_code != 200:
        return "Error: Unable to generate summary."

    return response.json().get("response", "")


def generate_flashcards(chunks):
    if not chunks:
        return "No content available for flashcards."

    joined_chunks = "\n\n".join(chunks)

    prompt = f"""
Create 5 study flashcards from the following content.
Format strictly as:
Q: ...
A: ...

Content:
{joined_chunks}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code != 200:
        return "Error: Unable to generate flashcards."

    return response.json().get("response", "")


if __name__ == "__main__":
    print("📚 Local RAG Study Assistant ready!")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("> ")

        if question.lower() in ["exit", "quit"]:
            break

        context = retrieve_context(question)
        answer = generate_answer(question, context)

        print("\nAnswer:")
        print(answer)
