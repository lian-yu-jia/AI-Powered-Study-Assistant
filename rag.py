import requests
from sentence_transformers import SentenceTransformer
from vectorstore import VectorStore
import streamlit as st


# Ollama config
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini"

@st.cache_resource
def load_vectorstore():
    return VectorStore.load("vectorstore.pkl")

# Load embedding model once
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()


def retrieve_context(question, top_k=3):
    vectorstore = load_vectorstore()
    query_embedding = embed_model.encode(question)
    results = vectorstore.query(query_embedding, top_k=top_k)
    return results

def generate_answer(question, context_chunks):
    
    if not context_chunks:
        return "I don't know"

    context = "\n\n".join(context_chunks)
    context = context[:3000]  # prevent token overflow
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

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=60
        )
        if response.status_code != 200:
            return f"Model Error ({response.status_code}): {response.text}"

        return response.json().get("response", "No response generated.")
    except Exception as e:
        return f"Error: {str(e)}"


# --- Helper to batch large chunks ---
def batch_chunks(chunks, batch_size=5):
    """Split list of chunks into smaller batches for faster processing."""
    for i in range(0, len(chunks), batch_size):
        yield chunks[i:i + batch_size]


