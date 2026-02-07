from ingest import extract_text_from_pdf
from preprocess import clean_text, chunk_text
from embedings import generate_embeddings
from vectorstore import VectorStore

# Step 1: extract + clean + chunk
text = extract_text_from_pdf("sample.pdf")
text = clean_text(text)
chunks = chunk_text(text)

# Step 2: generate embeddings
embeddings = generate_embeddings(chunks)

# Step 3: create vector store and add chunks
vs = VectorStore(embedding_dim=embeddings.shape[1])
vs.add(chunks, embeddings)

# Step 4: save vector store
vs.save("vectorstore.pkl")

# Step 5: load vector store and query
vs2 = VectorStore.load("vectorstore.pkl")

# Example query
query = "summarize key points about embeddings"
query_emb = generate_embeddings([query])[0]  # embedding of query
results = vs2.query(query_emb, top_k=3)

print("Top 3 retrieved chunks:")
for i, chunk in enumerate(results, 1):
    print(f"{i}: {chunk[:200]}...\n")  # show first 200 chars
