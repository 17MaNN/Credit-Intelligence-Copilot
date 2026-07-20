"""Request-ID generation and propagation, shared by every service.

Uses a contextvar rather than passing request_id through every function
call explicitly - once middleware sets it at the start of a request, any
code running within that request (including nested log calls and outbound
HTTP calls to other services) can read it back automatically. Safe under
threads (each thread gets its own view during Phase 13's concurrent tool
execution) and under asyncio (each request's handler gets its own context).

This is deliberately NOT full distributed tracing (no spans, no OpenTelemetry
collector) - it's the cheap, high-value version: grep one ID across all 5
services' logs and see a single request's full path through the system.
"""
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from fastapi import Request

_request_id: ContextVar[str] = ContextVar("request_id", default="-")

REQUEST_ID_HEADER = "X-Request-ID"


def get_request_id() -> str:
    """Returns the current request's ID, or '-' if called outside a
    request (e.g. at module import time)."""
    return _request_id.get()


@contextmanager
def use_request_id(req_id: str):
    """Sets the request ID for the duration of a `with` block, scoped to
    whichever thread calls this. Use this (NOT contextvars.copy_context()
    .run()) when propagating a request ID into a manually-spawned worker
    thread, e.g. inside a ThreadPoolExecutor: a single copied Context
    object cannot be entered concurrently by multiple threads at once
    (raises "cannot enter context: already entered" under real overlapping
    I/O), but each new OS thread already has its own separate default
    context, so a plain contextvar.set() from within that thread is safe."""
    token = _request_id.set(req_id)
    try:
        yield
    finally:
        _request_id.reset(token)


def add_request_id_middleware(app):
    """Attaches middleware that reads X-Request-ID from an inbound request
    if present (so a request already tagged by an upstream caller keeps
    its ID), otherwise generates a new one. Sets it on the contextvar for
    the duration of the request and echoes it back in the response header."""
    @app.middleware("http")
    async def _request_id_middleware(request: Request, call_next):
        incoming = request.headers.get(REQUEST_ID_HEADER)
        req_id = incoming if incoming else uuid.uuid4().hex[:12]
        token = _request_id.set(req_id)
        try:
            response = await call_next(request)
        finally:
            _request_id.reset(token)
        response.headers[REQUEST_ID_HEADER] = req_id
        return response