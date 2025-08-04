import faiss
import pickle
from sklearn.preprocessing import normalize

def retrieve_chunks(query, embedder, index_path, store_path, k):
    index = faiss.read_index(index_path)

    with open(store_path, "rb") as f:
        data = pickle.load(f)

    q_embed = embedder.encode([query], convert_to_numpy=True)
    q_embed = normalize(q_embed)

    D, I = index.search(q_embed, k * 10)

    results = []
    seen_paths = {}

    for idx in I[0]:
        item = data[idx]
        path = item.get("path")

        if path in seen_paths:
            seen_paths[path]["text"] += "\n" + item["text"]
        else:
            seen_paths[path] = item
            results.append(seen_paths[path])

        if len(results) >= k:
            break

    return results
