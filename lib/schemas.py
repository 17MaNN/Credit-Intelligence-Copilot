"""Shared request/response contracts. Every service imports from here —
no service redefines its own I/O shape from scratch."""
from pydantic import BaseModel


class ServiceResponse(BaseModel):
    ok: bool
    data: dict | None = None
    error: str | None = None
