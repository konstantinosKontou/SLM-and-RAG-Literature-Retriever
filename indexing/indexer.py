import os
import fitz  # PyMuPDF
import pickle
import faiss
from sklearn.preprocessing import normalize
from indexing.chunking import chunk_text_by_sentences  # Ensure this points to your function


def extract_metadata(doc):
    """
    Extracts title, author, and year from PDF metadata. Falls back to 'Unknown' if missing.
    """
    metadata = doc.metadata
    return {
        "title": metadata.get("title", "Unknown Title"),
        "author": metadata.get("author", "Unknown Author"),
        "year": metadata.get("creationDate", "Unknown Year")[2:6] if metadata.get("creationDate") else "Unknown Year"
    }


def index_pdfs(pdf_dir, embedder, index_path, store_path, chunk_size, overlap):
    """
    Indexes all PDFs in the given directory into a FAISS vector index using sentence-based chunking.
    """
    if os.path.exists(index_path) and os.path.exists(store_path):
        return

    texts, metadata = [], []
    first_pdf = sorted([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])[0]

    for fname in os.listdir(pdf_dir):
        if not fname.endswith(".pdf"):
            continue

        path = os.path.join(pdf_dir, fname)
        doc = fitz.open(path)
        doc_meta = extract_metadata(doc)

        # Combine all pages into one text blob
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"

        chunks = chunk_text_by_sentences(full_text, chunk_size=chunk_size, overlap_words=overlap)

        #Debug print for first PDF
        if fname == first_pdf:
            print(f"\nChunks from file: {fname}")
            for idx, c in enumerate(chunks):
                print(f"\n--- Chunk {idx + 1} ---")
                print(f"{c[:500]}...\nWords: {len(c.split())}")

        # Append chunks and metadata
        for chunk in chunks:
            texts.append(chunk)
            metadata.append({
                "source": fname,
                "page": "-",  # Can't assign exact page now (document-level chunking)
                "path": os.path.abspath(path),
                "title": doc_meta["title"],
                "author": doc_meta["author"],
                "year": doc_meta["year"]
            })