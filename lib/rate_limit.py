"""Simple in-memory rate limiter (fixed window). No external dependencies
(no Redis) - appropriate for a single-instance/demo deployment. Note: this
state is per-pod, so with multiple replicas each pod enforces its own limit
independently rather than a shared global limit. For a production multi-
replica deployment, swap this for a Redis-backed limiter instead."""
import time
import threading
from collections import defaultdict
from fastapi import HTTPException, Request


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, key: str):
        now = time.time()
        with self._lock:
            recent = [t for t in self._hits[key] if now - t < self.window_seconds]
            if len(recent) >= self.max_requests:
                self._hits[key] = recent
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: max {self.max_requests} requests "
                            f"per {self.window_seconds}s",
                )
            recent.append(now)
            self._hits[key] = recent


def make_rate_limit_dependency(limiter: RateLimiter):
    """Returns a FastAPI dependency that rate-limits by client IP."""
    def dependency(request: Request):
        client_key = request.client.host if request.client else "unknown"
        limiter.check(client_key)
    return dependency