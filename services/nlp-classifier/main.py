"""NLP intent classification service. Single endpoint: POST /classify-text.
Auth/schema/logging reused from lib/, same pattern as risk-model and doc-cv."""
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from lib.auth import verify_key
from lib.schemas import ServiceResponse
from lib.logging import get_logger
from data import LABELS

log = get_logger("nlp-classifier")
app = FastAPI(title="nlp-classifier")

MODEL_DIR = "nlp_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

MAX_CHARS = 2000  # basic input validation guard


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)


@app.get("/health")
def health():
    return {"ok": True}



@app.post("/classify-text", response_model=ServiceResponse)
def classify_text(req: ClassifyRequest, _=Depends(verify_key)):
    inputs = tokenizer(req.text, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax())

    log.info(f"classified text, intent={LABELS[pred_idx]}")
    return ServiceResponse(ok=True, data={
        "intent": LABELS[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
    })