from sentence_transformers import SentenceTransformer

def get_embedder(model_name):
    return SentenceTransformer(model_name)
