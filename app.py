import streamlit as st
from rag import generate_flashcards, generate_summary, retrieve_context, generate_answer
from ingest import extract_text_from_pdf
from preprocess import clean_text, chunk_text
from embedings import generate_embeddings
from vectorstore import VectorStore
import os


st.set_page_config(page_title="AI Study Assistant", layout="wide")

st.title("📚 AI-Powered Study Assistant")
st.caption("Local RAG-based document assistant (offline & free)")

# --- Sidebar ---
st.sidebar.header("Upload Document")
if "processed" not in st.session_state:
    st.session_state.processed = False

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.success("PDF uploaded successfully!")

    if st.sidebar.button("Process Document"):
        progress = st.progress(0)
        status = st.empty()

        status.text("Extracting text...")
        text = extract_text_from_pdf("temp.pdf")
        progress.progress(20)

        status.text("Cleaning text...")
        text = clean_text(text)
        progress.progress(40)

        status.text("Chunking document...")
        chunks = chunk_text(text)
        progress.progress(60)

        status.text("Generating embeddings...")
        embeddings = generate_embeddings(chunks)
        progress.progress(80)

        if embeddings is None or len(embeddings) == 0:
            st.error("Failed to generate embeddings.")
            st.stop()

        status.text("Building vector database...")
        vs = VectorStore(embedding_dim=len(embeddings[0]))
        vs.add(chunks, embeddings)
        vs.save("vectorstore.pkl")
        progress.progress(100)

        st.session_state.processed = True
        status.text("Done!")
        st.sidebar.success("Document processed and indexed!")


st.divider()

# --- Q&A Section ---
st.header("Ask a question")

question = st.text_input("Enter your question")

if st.button("Ask", disabled=not st.session_state.processed):
    if not st.session_state.processed:
        st.warning("Please upload and process a document first.")
        st.stop()

    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            context = retrieve_context(question)
            answer = generate_answer(question, context)

        st.subheader("Answer")
        st.write(answer)

if st.button("Generate Summary", disabled=not st.session_state.processed):
    if not st.session_state.processed:
        st.warning("Please upload and process a document first.")
        st.stop()

    with st.spinner("Summarizing..."):
        chunks = retrieve_context("Summarize the document", top_k=3)
        summary = generate_summary(chunks)

    st.subheader("Document Summary")
    st.write(summary)


if st.button("Generate Flashcards", disabled=not st.session_state.processed):
    if not st.session_state.processed:
        st.warning("Please upload and process a document first.")
        st.stop()

    with st.spinner("Generating flashcards..."):
        chunks = retrieve_context("Generate flashcards", top_k=3)
        flashcards = generate_flashcards(chunks)

    st.subheader("Flashcards")
    st.text(flashcards)
