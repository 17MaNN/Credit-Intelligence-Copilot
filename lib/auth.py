"""Single source of truth for service-to-service auth.
Key is injected via K8s Secret env var, never hardcoded."""
import os
from fastapi import Header, HTTPException

SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY", "")


def verify_key(x_api_key: str = Header(default="")):
    if not SERVICE_API_KEY or x_api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")
