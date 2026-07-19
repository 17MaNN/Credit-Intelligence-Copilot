"""Agent orchestrator service. Two endpoints for the same logic, different
trust boundaries:
  - POST /handle-inquiry: requires x-api-key. For service-to-service calls
    (the eval harness, CI, future internal callers).
  - POST /ui/handle-inquiry: no API key. For the browser UI served from
    static/ below. The browser can't hold a real secret (anything in
    client-side JS is visible in page source), so this endpoint relies on
    rate limiting alone, same as any public-facing demo endpoint would.
Auth/schema/logging reused from lib/, same pattern as all prior services.
This is the only service that talks to an LLM (Gemini) - everything below
it (risk-model, doc-cv, nlp-classifier, rag) is a plain, secure HTTP tool."""
import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lib.auth import verify_key
from lib.schemas import ServiceResponse
from lib.logging import get_logger
from lib.rate_limit import RateLimiter, make_rate_limit_dependency
from gemini_client import run_agent
from workflow import build_workflow

log = get_logger("agent")
app = FastAPI(title="agent")

# Browser calls to /ui/handle-inquiry come from the static UI served below.
# CORS_ALLOWED_ORIGINS lets you restrict this in production; defaults to
# allow-all for local development where the UI may be opened from a
# file:// URL or a different port.
ALLOWED_ORIGINS = os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["x-api-key", "Content-Type"],
)

# Protects the Gemini quota and the service itself from abuse - each caller
# IP gets 10 requests per 60s. Applied to both endpoints below.
inquiry_limiter = RateLimiter(max_requests=10, window_seconds=60)
rate_limit = make_rate_limit_dependency(inquiry_limiter)


class InquiryRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    # Optional base64-encoded image attachment. doc-cv enforces an 8MB raw
    # upload limit; base64 encoding inflates size by ~33%, so this cap
    # gives an early, cheap rejection before the payload travels any
    # further through the pipeline.
    image_base64: str | None = Field(default=None, max_length=11_000_000)


@app.get("/health")
def health():
    return {"ok": True}


def _process_inquiry(message: str, image_base64: str | None = None) -> ServiceResponse:
    """Shared core logic for both endpoints below - only the auth
    requirement differs between them."""
    try:
        final_text, tool_call_log = run_agent(message, image_base64=image_base64)
    except Exception as e:
        log.info(f"agent failed: {e}")
        return ServiceResponse(ok=False, error=f"agent temporarily unavailable: {e}")

    log.info(f"handled inquiry, tool_calls={len(tool_call_log)}")
    return ServiceResponse(ok=True, data={
        "response": final_text,
        "workflow": build_workflow(tool_call_log),  # human-readable step-by-step view
        "tool_calls": tool_call_log,  # raw audit trail (kept for debugging/eval)
    })


@app.post("/handle-inquiry", response_model=ServiceResponse)
def handle_inquiry(req: InquiryRequest, _auth=Depends(verify_key), _rl=Depends(rate_limit)):
    return _process_inquiry(req.message, req.image_base64)


@app.post("/ui/handle-inquiry", response_model=ServiceResponse)
def ui_handle_inquiry(req: InquiryRequest, _rl=Depends(rate_limit)):
    return _process_inquiry(req.message, req.image_base64)

# Mounted last so it doesn't shadow the API routes above. Serves the web UI
# at the root path.
app.mount("/", StaticFiles(directory="static", html=True), name="static")