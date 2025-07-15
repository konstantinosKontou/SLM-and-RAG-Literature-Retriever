from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_document(text: str, chunk_size, overlap_words):
    """
    Chunks text using LangChain's RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # characters, not words
        chunk_overlap=overlap_words,  # also in characters
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_text(text)