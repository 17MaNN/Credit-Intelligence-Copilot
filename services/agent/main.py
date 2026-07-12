"""Agent orchestrator service. Single endpoint: POST /handle-inquiry.
Auth/schema/logging reused from lib/, same pattern as all prior services.
This is the only service that talks to an LLM - everything below it
(risk-model, doc-cv, nlp-classifier, rag) is a plain, secure HTTP tool."""
from gemini_client import run_agent
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field

from lib.auth import verify_key
from lib.schemas import ServiceResponse
from lib.logging import get_logger

log = get_logger("agent")
app = FastAPI(title="agent")


class InquiryRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/handle-inquiry", response_model=ServiceResponse)
def handle_inquiry(req: InquiryRequest, _=Depends(verify_key)):
    final_text, tool_call_log = run_agent(req.message)

    log.info(f"handled inquiry, tool_calls={len(tool_call_log)}")
    return ServiceResponse(ok=True, data={
        "response": final_text,
        "tool_calls": tool_call_log,  # audit trail
    })