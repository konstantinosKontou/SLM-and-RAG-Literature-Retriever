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
        print("Index and store already exist. Skipping.")
        return

    texts, metadata = [] , []
    pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])

    if not pdf_files:
        print("No PDF files found.")
        return

    for fname in pdf_files:
        path = os.path.join(pdf_dir, fname)
        print(f"\n Processing {fname}")
        doc = fitz.open(path)
        doc_meta = extract_metadata(doc)

        # LOGGING WORDS FOR DEBUG
        # total_words = 0
        # for i, page in enumerate(doc):
        #     page_text = page.get_text()
        #     word_count = len(page_text.split())
        #     total_words += word_count
        #     print(f"Page {i+1}: {word_count} words")

        # COMBINE PAGES INTO ONE TEXT
        full_text = "\n".join(page.get_text() for page in doc)

        # CLEAN AND TRIM THE TEXT
        cleaned_text = clean_text(full_text)

        # REMOVE REFERENCE LIST
        cleaned_text = remove_references_section(cleaned_text)

        # CHUNKING
        chunks = chunk_document(cleaned_text, chunk_size, overlap)

        # for idx, chunk in enumerate(chunks):
        #     print(f"\n--- Chunk {idx + 1} ---")
        #     print(chunk)
        #     print(f"\nWords: {len(chunk.split())}")
        #

        for chunk in chunks:
            texts.append(chunk)
            metadata.append({
                "source": fname,
                "path": os.path.abspath(path),
                "title": doc_meta["title"],
                "author": doc_meta["author"],
                "year": doc_meta["year"]
            })