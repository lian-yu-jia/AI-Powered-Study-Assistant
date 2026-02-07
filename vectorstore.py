import faiss
import numpy as np
import pickle

class VectorStore:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
        self.metadata = []  # store text chunks

    def add(self, chunks, embeddings):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.metadata.extend(chunks)

    def save(self, path="vectorstore.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"index": self.index, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, path="vectorstore.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        vs = cls(embedding_dim=data["index"].d)
        vs.index = data["index"]
        vs.metadata = data["metadata"]
        return vs

    def query(self, query_embedding, top_k=5):
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)
        results = [self.metadata[i] for i in indices[0]]
        return results
