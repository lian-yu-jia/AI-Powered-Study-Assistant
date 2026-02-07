from ingest import extract_text_from_pdf
from preprocess import clean_text, chunk_text
from embedings import generate_embeddings

text = extract_text_from_pdf("sample.pdf")
text = clean_text(text)
chunks = chunk_text(text)

embeddings = generate_embeddings(chunks)
print("Embeddings shape:", embeddings.shape)
