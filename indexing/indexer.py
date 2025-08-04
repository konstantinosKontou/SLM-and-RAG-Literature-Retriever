import streamlit as st
import os
import fitz  # PyMuPDF
import pickle
import faiss
from sklearn.preprocessing import normalize
from indexing.chunking import chunk_document

# EXTRACT METADATA FOR EACH DOCUMENT
def extract_metadata(doc):
    metadata = doc.metadata
    creation_date = metadata.get("creationDate")
    year = creation_date[2:6] if creation_date and len(creation_date) >= 6 else "Unknown Year"

    return {
        "title": metadata.get("title", "Unknown Title"),
        "author": metadata.get("author", "Unknown Author"),
        "year": year
    }


def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if (
            line.lower().startswith("research in social and administrative pharmacy")
            or line.lower().startswith("available online")
            or line.lower().startswith("©")
            or line.lower().startswith("doi:")
            or line.lower().startswith("https://")
            or line.lower().startswith("http://")
            or line == ""
        ):
            continue
        cleaned_lines.append(line)

    return " ".join(cleaned_lines)


def remove_references_section(text: str) -> str:
    import re
    match = re.search(r"\bReferences\b", text, re.IGNORECASE)
    return text[:match.start()] if match else text


def index_pdfs(pdf_dir, embedder, index_path, store_path, chunk_size, overlap):
    if os.path.exists(index_path) and os.path.exists(store_path):
        st.info("Index and store already exist.")
        return

    all_chunks = []
    pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])

    if not pdf_files:
        return

    for fname in pdf_files:
        path = os.path.join(pdf_dir, fname)
        doc = fitz.open(path)
        doc_meta = extract_metadata(doc)

        # COMBINE PAGES INTO ONE TEXT
        full_text = "\n".join(page.get_text() for page in doc)

        # CLEAN AND TRIM THE TEXT
        cleaned_text = clean_text(full_text)

        # REMOVE REFERENCE LIST
        cleaned_text = remove_references_section(cleaned_text)

        # CHUNKING
        chunks = chunk_document(cleaned_text, chunk_size, overlap)

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": fname,
                "path": os.path.abspath(path),
                "title": doc_meta["title"],
                "author": doc_meta["author"],
                "year": doc_meta["year"]
            })

    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    embeddings = normalize(embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(store_path), exist_ok=True)

    # SAVE INDEX AND CHUNK
    faiss.write_index(index, index_path)
    with open(store_path, "wb") as f:
        pickle.dump(all_chunks, f)