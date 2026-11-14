# simple rag pipeline demo for embedding and text processing
from sentence_transformers import SentenceTransformer

def generate_embedding(text: str):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode(text)
    return embedding

if __name__ == "__main__":
    sample_text = "document retrieval for LLM application"
    emb = generate_embedding(sample_text)
    print(emb.shape)
