import streamlit as st
from rag import retrieve_context, generate_answer
from summarizer import summarize_document
from flashcards import parse_flashcards
from preprocess import clean_text, chunk_text
from ingest import extract_text_from_pdf
from vectorstore import VectorStore
from sentence_transformers import SentenceTransformer
import os
import streamlit as st
import time

# -------------------
# Embedding Model
# -------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

# -------------------
# Page Config
# -------------------
st.set_page_config(page_title="StudyAI", page_icon="🧠", layout="wide")

# -------------------
# Styling
# -------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f6f9fc 0%, #eef2f7 100%);
}
.card {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}
.stButton>button {
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-weight: 500;
    border: none;
    background: linear-gradient(90deg, #4f8cff, #6a5acd);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------
# Session State
# -------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "summary" not in st.session_state:
    st.session_state.summary = None


# =============================
# HEADER
# =============================
st.title("🧠 StudyAI")
st.caption("Your AI-powered document intelligence assistant")


# =============================
# UPLOAD SECTION
# =============================
st.subheader("📤 Upload & Index Documents")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key="upload_main"
)

# Add an explicit indexing button
if st.button("Index Documents"):

    if uploaded_files:
        with st.spinner("Indexing documents... Please wait 🧠"):

            all_chunks = []

            for uploaded_file in uploaded_files:
                text = extract_text_from_pdf(uploaded_file)
                cleaned = clean_text(text)
                chunks = chunk_text(cleaned)
                all_chunks.extend(chunks)

            if all_chunks:
                embeddings = embed_model.encode(all_chunks)

                vs = VectorStore(len(embeddings[0]))
                vs.add(all_chunks, embeddings)
                vs = VectorStore.load("vectorstore.pkl")

                st.session_state.chunks = all_chunks
                st.session_state.vector_ready = True
                st.session_state.summary = None   # reset summary if new doc
                st.session_state.flashcards = []

        st.success("✅ Documents indexed successfully!")

    else:
        st.warning("Please upload files first.")

# =============================
# MAIN FUNCTIONALITY TABS
# =============================
selected_tab = st.radio(
    "",
    ["💬 Ask AI", "📝 Summary", "🗂 Flashcards"],
    horizontal=True
)
# ---------------------------------
# ASK AI TAB
# ---------------------------------
if selected_tab == "💬 Ask AI":
    st.subheader("Ask questions about your documents")

    user_input = st.text_input("Enter your question")

    if st.button("Generate Answer") and user_input:
        if st.session_state.vector_ready:
            with st.spinner("AI is thinking... 🧠"):
                context = retrieve_context(user_input, top_k=5)
                answer = generate_answer(user_input, context)

                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", answer))

        else:
            st.warning("Please upload and index a document first.")

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

# ---------------------------------
# SUMMARY TAB
# ---------------------------------
elif selected_tab == "📝 Summary":
    st.subheader("Generate Executive Summary")

    if st.button("Generate Summary"):

        if st.session_state.chunks:

            try:
                with st.spinner("Generating summary..."):
                    st.session_state.summary = summarize_document(st.session_state.chunks)

            except Exception as e:
                st.error("⚠️ Document set too large to summarize at once.")
                st.info("Try uploading fewer documents or shorter PDFs.")
                st.session_state.summary = None
        else:
            st.warning("Please index documents first.")

    if st.session_state.summary:
        st.write(st.session_state.summary)

# ---------------------------------
# FLASHCARDS TAB
# ---------------------------------

elif selected_tab == "🗂 Flashcards":
    st.subheader("Auto-Generated Study Flashcards")

    if st.button("Generate Flashcards"):
        
        from flashcards import generate_flashcards_from_summary

        if st.session_state.chunks:
            # Only generate summary if we don't already have one
            if st.session_state.summary is None:
                with st.spinner("Generating summary first..."):
                    st.session_state.summary = summarize_document(st.session_state.chunks)

            with st.spinner("Creating flashcards from summary..."):
                raw = generate_flashcards_from_summary(st.session_state.summary)
                cards = parse_flashcards(raw)
                st.session_state.flashcards = cards
        else:
            st.warning("Please upload a document first.")

    # Display flashcards
    if st.session_state.flashcards:
        cols = st.columns(2)
        for i, (q, a) in enumerate(st.session_state.flashcards):
            col = cols[i % 2]
            with col:
                with st.expander(q):  # Question becomes the clickable title
                    st.markdown("### Answer")
                    st.markdown(a)