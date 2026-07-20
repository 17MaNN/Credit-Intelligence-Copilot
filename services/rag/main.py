"""RAG retrieval service. Single endpoint: POST /retrieve.
Pure retrieval - no LLM here, kept separate so it scales independently of the
agent orchestrator (Phase 5). Auth/schema/logging reused from lib/."""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from lib.request_id import add_request_id_middleware
from lib.auth import verify_key
from lib.schemas import ServiceResponse
from lib.logging import get_logger

log = get_logger("rag")
app = FastAPI(title="rag")
add_request_id_middleware(app)

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index("index.faiss")
with open("chunks.json") as f:
    CHUNKS = json.load(f)

TOP_K_MAX = 10


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=TOP_K_MAX)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/retrieve", response_model=ServiceResponse)
def retrieve(req: RetrieveRequest, _=Depends(verify_key)):
    q_emb = model.encode([req.query], normalize_embeddings=True)
    scores, idxs = index.search(np.array(q_emb, dtype=np.float32), req.top_k)

    results = [
        {"id": CHUNKS[i]["id"], "text": CHUNKS[i]["text"], "score": round(float(s), 4)}
        for s, i in zip(scores[0], idxs[0]) if i != -1
    ]

    log.info(f"retrieved {len(results)} chunks for query")
    return ServiceResponse(ok=True, data={"results": results})