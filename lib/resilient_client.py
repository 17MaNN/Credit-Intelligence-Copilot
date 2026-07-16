"""Resilient HTTP client: retry-with-backoff + circuit breaker, reused by
every backend call the agent makes (tools.py). Generalizes the same retry
pattern already proven in gemini_client.py's _send_with_retry.

Circuit breaker: after MAX_FAILURES consecutive failures against a given
host, the circuit "opens" and fails fast (no network call) for
COOLDOWN_SECONDS, instead of repeatedly waiting on a timeout against a
service that's confirmed down. After the cooldown, one "trial" request is
allowed through (half-open); success closes the circuit, failure re-opens it.
"""
import time
import threading
import httpx

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 0.5
MAX_FAILURES = 5
COOLDOWN_SECONDS = 30


class CircuitOpenError(Exception):
    pass


class CircuitBreaker:
    def __init__(self, max_failures=MAX_FAILURES, cooldown_seconds=COOLDOWN_SECONDS):
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        self._state = {}  # key -> {"failures": int, "opened_at": float|None}
        self._lock = threading.Lock()

    def _get(self, key):
        return self._state.setdefault(key, {"failures": 0, "opened_at": None})

    def before_request(self, key):
        """Raises CircuitOpenError if the circuit is open and cooldown hasn't
        passed. Otherwise allows the request through (including the single
        half-open trial after cooldown)."""
        with self._lock:
            s = self._get(key)
            if s["opened_at"] is not None:
                elapsed = time.time() - s["opened_at"]
                if elapsed < self.cooldown_seconds:
                    raise CircuitOpenError(
                        f"circuit open for {key}, retry in {self.cooldown_seconds - elapsed:.0f}s"
                    )
                # cooldown passed - allow one half-open trial through

    def record_success(self, key):
        with self._lock:
            self._state[key] = {"failures": 0, "opened_at": None}

    def record_failure(self, key):
        with self._lock:
            s = self._get(key)
            s["failures"] += 1
            if s["failures"] >= self.max_failures:
                s["opened_at"] = time.time()


# Module-level shared breaker so state persists across calls within a process
_breaker = CircuitBreaker()


def resilient_post(url, json=None, files=None, headers=None, timeout=15.0):
    """POST with retry-with-backoff on transient errors, gated by a circuit
    breaker keyed on the target host. Raises on failure - callers (tools.py)
    already handle exceptions gracefully by returning them as tool results."""
    key = httpx.URL(url).host

    _breaker.before_request(key)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            r = httpx.post(url, json=json, files=files, headers=headers, timeout=timeout)
            if r.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"server error {r.status_code}", request=r.request, response=r
                )
            r.raise_for_status()  # 4xx - not retryable, raises immediately below
            _breaker.record_success(key)
            return r
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code < 500:
                raise  # client error - retrying won't help
            last_error = e
        except httpx.RequestError as e:
            last_error = e

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))

    _breaker.record_failure(key)
    raise last_error