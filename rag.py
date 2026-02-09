import requests
from sentence_transformers import SentenceTransformer
from vectorstore import VectorStore

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini"

# Load embedding model (same as Week 1)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load vector store
vectorstore = VectorStore.load("vectorstore.pkl")


def retrieve_context(question, top_k=5):
    query_embedding = embed_model.encode(question)
    return vectorstore.query(query_embedding, top_k=top_k)


def generate_answer(question, context_chunks):
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

    # If Ollama itself failed
    if response.status_code != 200:
        return f"[Ollama HTTP error]: {response.text}"

    try:
        data = response.json()
    except Exception:
        return f"[Invalid JSON from Ollama]: {response.text}"

    # DEBUG (keep for now)
    print("\n[DEBUG] Ollama response:", data)

    # Safe access
    return data.get("response", "[No response returned by model]")


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
