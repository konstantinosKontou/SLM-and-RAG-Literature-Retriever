import re

def chunk_text_by_sentences(text, chunk_size=500, overlap_words=50):
    """
    Splits text into sentence-based chunks of approximately `chunk_size` words,
    with `overlap_words` for context retention.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) > chunk_size:
            chunks.append(" ".join(current_chunk))
            # Add overlap
            if overlap_words > 0:
                flat = " ".join(current_chunk).split()
                overlap_chunk = flat[-overlap_words:] if len(flat) >= overlap_words else flat
                current_chunk = [" ".join(overlap_chunk)]
                word_count = len(overlap_chunk)
            else:
                current_chunk = []
                word_count = 0
        current_chunk.append(sentence)
        word_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
