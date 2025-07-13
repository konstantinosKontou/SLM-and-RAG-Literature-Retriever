import os
import streamlit as st
from indexing.indexer import index_pdfs
from indexing.embedder import get_embedder

from config import EMBED_MODEL_NAME, INDEX_PATH, DOC_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP

# SIDEBAR
st.sidebar.title("Settings")
pdf_folder = st.sidebar.text_input("PDF Folder Path", value="papers/")
top_k = st.sidebar.slider("Top K Chunks to Retrieve", 1, 10, 3)

# CHAT SECTION
st.title("Local Academic RAG Assistant")
st.markdown("Ask a question about your local academic PDFs.")
user_question = st.text_input("Ask your question:")
ask_button = st.button("Submit Question")

if ask_button and user_question:

    # CHECK IF THE PATH EXISTS
    # IF NOT EXISTS
    if not os.path.exists(pdf_folder):
        st.error(f"Folder '{pdf_folder}' not found.")

    # IF EXISTS
    else:
        st.success(f"Folder '{pdf_folder}' found.")

        # PROCEED WITH THE INDEXING OF THE DOCUMENTS (PDF FROM LOCAL FOLDER)
        st.info("Indexing documents (if not already indexed)...")
        embedder = get_embedder(EMBED_MODEL_NAME)
        index_pdfs(pdf_folder, embedder, INDEX_PATH, DOC_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
