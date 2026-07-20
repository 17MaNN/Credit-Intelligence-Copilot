"""Lightweight in-memory metrics, exposed in Prometheus text exposition
format via a /metrics endpoint. No Prometheus/Grafana server is deployed
anywhere in this project - this just makes the service scrape-compatible
if one is ever added, without taking on that infrastructure now.

State resets on restart and is per-pod (same limitation as
lib/rate_limit.py) - acceptable for this project's scope; a real
multi-replica production deployment would centralize this instead.
"""
import threading
from collections import defaultdict

_lock = threading.Lock()
_counters = defaultdict(int)
_latency_sum_ms = defaultdict(float)
_latency_count = defaultdict(int)


def inc(name: str, labels: dict = None, amount: int = 1):
    """Increments a named counter, optionally with labels (e.g.
    tool='classify_intent', status='success')."""
    with _lock:
        _counters[_key(name, labels)] += amount


def observe_latency(name: str, ms: float, labels: dict = None):
    """Records a latency observation. Exposed as sum+count (average is
    sum/count) rather than a full histogram - enough to spot regressions
    without the complexity of real histogram buckets."""
    key = _key(name, labels)
    with _lock:
        _latency_sum_ms[key] += ms
        _latency_count[key] += 1


def _key(name, labels):
    if not labels:
        return (name, ())
    return (name, tuple(sorted(labels.items())))


def _format_labels(label_tuple):
    if not label_tuple:
        return ""
    parts = ",".join(f'{k}="{v}"' for k, v in label_tuple)
    return "{" + parts + "}"


def render_prometheus_text() -> str:
    """Renders all current metrics in Prometheus text exposition format."""
    lines = []
    with _lock:
        for (name, labels), value in sorted(_counters.items()):
            lines.append(f"{name}{_format_labels(labels)} {value}")
        for (name, labels), total_ms in sorted(_latency_sum_ms.items()):
            count = _latency_count[(name, labels)]
            avg_ms = total_ms / count if count else 0
            lines.append(f"{name}_avg_ms{_format_labels(labels)} {avg_ms:.2f}")
            lines.append(f"{name}_count{_format_labels(labels)} {count}")
    return "\n".join(lines) + "\n"