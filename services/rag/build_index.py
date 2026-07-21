"""Embed policy docs and build a FAISS index. Run once at build time: python build_index.py
Requires internet access to download the embedding model from Hugging Face Hub -
this happens during `docker build`, not at runtime."""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from docs import DOCS

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "index.faiss"
CHUNKS_PATH = "chunks.json"

model = SentenceTransformer(MODEL_NAME)
texts = [d["text"] for d in DOCS]
embeddings = model.encode(texts, normalize_embeddings=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
index.add(np.array(embeddings, dtype=np.float32))

faiss.write_index(index, INDEX_PATH)
with open(CHUNKS_PATH, "w") as f:
    json.dump(DOCS, f)

print(f"indexed {len(DOCS)} docs, dim={dim}")