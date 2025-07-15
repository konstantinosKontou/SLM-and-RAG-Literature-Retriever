import faiss
import pickle
from sklearn.preprocessing import normalize

def retrieve_chunks(query, embedder, index_path, store_path, k):
    index = faiss.read_index(index_path)

    with open(store_path, "rb") as f:
        data = pickle.load(f)

    q_embed = embedder.encode([query], convert_to_numpy=True)
    q_embed = normalize(q_embed)


    D, I = index.search(q_embed, k * 5)

    results = []
    seen_sources = set()

    for idx in I[0]:
        item = data[idx]
        source = item.get("source")
        if source not in seen_sources:
            seen_sources.add(source)
            results.append(item)
        if len(results) >= k:
            break

    return results