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
it (risk-model, doc-cv, nlp-classifier, rag) is a plain, secure HTTP tool.

Observability (Phase 15): every request gets an ID (auto-generated or
carried from an upstream caller) that shows up in this service's logs AND
gets forwarded to the 4 backend services, so a single grep for one ID
across all 5 services' logs shows that request's full path through the
system. /metrics exposes basic counters in Prometheus text format - no
Prometheus/Grafana server is actually deployed, this just makes the
service scrape-compatible if one is ever added."""
import os
import json
import time
from fastapi import FastAPI, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lib.auth import verify_key
from lib.schemas import ServiceResponse
from lib.logging import get_logger
from lib.rate_limit import RateLimiter, make_rate_limit_dependency
from lib.request_id import add_request_id_middleware, get_request_id
from lib import metrics
from gemini_client import run_agent, run_agent_stream
from workflow import build_workflow

log = get_logger("agent")
app = FastAPI(title="agent")
add_request_id_middleware(app)

ALLOWED_ORIGINS = os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["x-api-key", "Content-Type"],
)

inquiry_limiter = RateLimiter(max_requests=10, window_seconds=60)
rate_limit = make_rate_limit_dependency(inquiry_limiter)


class InquiryRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    image_base64: str | None = Field(default=None, max_length=11_000_000)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/metrics")
def get_metrics():
    return Response(content=metrics.render_prometheus_text(), media_type="text/plain")


def _process_inquiry(message: str, image_base64: str | None = None) -> ServiceResponse:
    start = time.monotonic()
    metrics.inc("agent_requests_total")

    try:
        final_text, tool_call_log = run_agent(message, image_base64=image_base64)
    except Exception as e:
        log.info(f"agent failed: {e}")
        metrics.inc("agent_requests_total", {"status": "error"})
        metrics.observe_latency("agent_request_latency_ms", (time.monotonic() - start) * 1000)
        return ServiceResponse(ok=False, error=f"agent temporarily unavailable: {e}")

    for call in tool_call_log:
        is_error = isinstance(call.get("result"), dict) and (
            call["result"].get("error") or call["result"].get("ok") is False
        )
        metrics.inc("tool_calls_total", {
            "tool": call["tool"],
            "status": "error" if is_error else "success",
        })

    metrics.inc("agent_requests_total", {"status": "success"})
    metrics.observe_latency("agent_request_latency_ms", (time.monotonic() - start) * 1000)

    log.info(f"handled inquiry, tool_calls={len(tool_call_log)}")
    return ServiceResponse(ok=True, data={
        "response": final_text,
        "workflow": build_workflow(tool_call_log),
        "tool_calls": tool_call_log,
    })


@app.post("/handle-inquiry", response_model=ServiceResponse)
def handle_inquiry(req: InquiryRequest, _auth=Depends(verify_key), _rl=Depends(rate_limit)):
    return _process_inquiry(req.message, req.image_base64)


@app.post("/ui/handle-inquiry", response_model=ServiceResponse)
def ui_handle_inquiry(req: InquiryRequest, _rl=Depends(rate_limit)):
    return _process_inquiry(req.message, req.image_base64)


@app.post("/ui/handle-inquiry/stream")
def ui_handle_inquiry_stream(req: InquiryRequest, _rl=Depends(rate_limit)):
    """Server-Sent Events variant of /ui/handle-inquiry, for the browser UI
    only - /handle-inquiry (used by the eval harness, CI) is untouched."""
    def event_generator():
        start = time.monotonic()
        metrics.inc("agent_requests_total", {"mode": "stream"})
        try:
            for event in run_agent_stream(req.message, image_base64=req.image_base64):
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] == "tool_end":
                    metrics.inc("tool_calls_total", {"tool": event["tool"], "status": event["status"]})
                if event["type"] == "done":
                    metrics.inc("agent_requests_total", {"status": "success", "mode": "stream"})
                    metrics.observe_latency(
                        "agent_request_latency_ms", (time.monotonic() - start) * 1000, {"mode": "stream"}
                    )
                    log.info(f"handled inquiry (stream), tool_calls={len(event['tool_calls'])}")
        except Exception as e:
            log.info(f"agent stream failed: {e}")
            metrics.inc("agent_requests_total", {"status": "error", "mode": "stream"})
            metrics.observe_latency(
                "agent_request_latency_ms", (time.monotonic() - start) * 1000, {"mode": "stream"}
            )
            yield f"data: {json.dumps({'type': 'error', 'error': f'agent temporarily unavailable: {e}'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


app.mount("/", StaticFiles(directory="static", html=True), name="static")