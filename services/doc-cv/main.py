"""Document CV service. Single endpoint: POST /analyze-document (multipart image upload).
Returns doc-type classification + OCR'd text. Auth/schema/logging reused from lib/,
weights_only=True on torch.load per security principle (see risk-model fix)."""
import io
import torch
import pytesseract
from PIL import Image
from fastapi import FastAPI, Depends, UploadFile, File
import torch.nn.functional as F

from lib.auth import verify_key
from lib.schemas import ServiceResponse
from lib.logging import get_logger
from model import DocNet, N_CLASSES
from data import CLASSES, IMG_SIZE

log = get_logger("doc-cv")
app = FastAPI(title="doc-cv")

model = DocNet()
model.load_state_dict(torch.load("doc_model.pt", map_location="cpu", weights_only=True))
model.eval()

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8MB cap - basic input validation / DoS guard


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze-document", response_model=ServiceResponse)
async def analyze_document(file: UploadFile = File(...), _=Depends(verify_key)):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        return ServiceResponse(ok=False, error="file too large")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return ServiceResponse(ok=False, error="invalid image")

    # doc-type classification
    resized = img.resize((IMG_SIZE, IMG_SIZE))
    x = torch.tensor(list(resized.getdata()), dtype=torch.float32)
    x = x.view(IMG_SIZE, IMG_SIZE, 3).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax())

    # OCR
    text = pytesseract.image_to_string(img).strip()

    log.info(f"analyzed doc, type={CLASSES[pred_idx]}, ocr_chars={len(text)}")
    return ServiceResponse(ok=True, data={
        "doc_type": CLASSES[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "extracted_text": text,
    })