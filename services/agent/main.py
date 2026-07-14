"""Agent orchestrator service. Single endpoint: POST /handle-inquiry.
Auth/schema/logging reused from lib/, same pattern as all prior services.
This is the only service that talks to an LLM (Gemini) - everything below
it (risk-model, doc-cv, nlp-classifier, rag) is a plain, secure HTTP tool."""
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field

from lib.auth import verify_key
from lib.schemas import ServiceResponse
from lib.logging import get_logger
from lib.rate_limit import RateLimiter, make_rate_limit_dependency
from gemini_client import run_agent

log = get_logger("agent")
app = FastAPI(title="agent")

# Protects the Gemini quota and the service itself from abuse - each caller
# IP gets 10 requests per 60s. Adjust to fit expected real usage.
inquiry_limiter = RateLimiter(max_requests=10, window_seconds=60)
rate_limit = make_rate_limit_dependency(inquiry_limiter)


class InquiryRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/handle-inquiry", response_model=ServiceResponse)
def handle_inquiry(req: InquiryRequest, _auth=Depends(verify_key), _rl=Depends(rate_limit)):
    try:
        final_text, tool_call_log = run_agent(req.message)
    except Exception as e:
        log.info(f"agent failed: {e}")
        return ServiceResponse(ok=False, error=f"agent temporarily unavailable: {e}")

    log.info(f"handled inquiry, tool_calls={len(tool_call_log)}")
    return ServiceResponse(ok=True, data={
        "response": final_text,
        "tool_calls": tool_call_log,  # audit trail
    })