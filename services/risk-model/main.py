"""Credit risk scoring service. Single endpoint: POST /predict.
Auth, response shape, and logging all come from lib/ - not redefined here."""
import json
import torch
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field

from lib.auth import verify_key
from lib.schemas import ServiceResponse
from lib.logging import get_logger
from model import RiskNet

log = get_logger("risk-model")
app = FastAPI(title="risk-model")

model = RiskNet()
model.load_state_dict(torch.load("risk_model.pt", map_location="cpu", weights_only=True))
model.eval()

with open("norm_stats.json") as f:
    stats = json.load(f)
MEAN = torch.tensor(stats["mean"])
STD = torch.tensor(stats["std"])


class RiskRequest(BaseModel):
    income: float = Field(gt=0)
    debt_to_income: float = Field(ge=0, le=1)
    credit_history_years: float = Field(ge=0)
    num_delinquencies: int = Field(ge=0)
    loan_amount: float = Field(gt=0)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict", response_model=ServiceResponse)
def predict(req: RiskRequest, _=Depends(verify_key)):
    x = torch.tensor([[req.income, req.debt_to_income, req.credit_history_years,
                        req.num_delinquencies, req.loan_amount]], dtype=torch.float32)
    x_norm = (x - MEAN) / STD
    with torch.no_grad():
        score = model(x_norm).item()

    log.info(f"scored request, default_probability={score:.4f}")
    return ServiceResponse(ok=True, data={
        "default_probability": round(score, 4),
        "risk_band": "high" if score > 0.6 else "medium" if score > 0.3 else "low",
    })