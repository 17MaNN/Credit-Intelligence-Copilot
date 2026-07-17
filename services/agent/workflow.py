"""Translates the raw tool-call audit trail into a human-readable workflow
view: what the system did, in plain language, not the model's internal
reasoning (which isn't exposed here anyway - thinking is disabled in
gemini_client.py). This is presentation logic specific to the agent
service's response shape, not shared plumbing, so it lives here rather
than in lib/."""

STEP_LABELS = {
    "classify_intent": "Detecting customer intent",
    "get_credit_risk": "Evaluating credit risk",
    "retrieve_policy": "Retrieving relevant lending policy",
    "analyze_document": "Processing uploaded document",
}


def _is_success(result):
    """Two shapes reach here: a full ServiceResponse dict from a backend
    call ({"ok": true/false, "data": ..., "error": null-or-message}), or a
    bare {"error": "<exception message>"} from gemini_client.py catching a
    tool execution failure (circuit breaker, connection error, etc). In
    both cases a truthy "error" value means failure - its mere presence as
    a key does not, since ServiceResponse always includes the key."""
    if not isinstance(result, dict):
        return False
    if result.get("error"):
        return False
    if result.get("ok") is False:
        return False
    return True


def _summarize(tool_name, result):
    """Short plain-language detail for a successful step. Returns None if
    there's nothing meaningful to summarize (falls back to just the label)."""
    data = result.get("data") if isinstance(result, dict) else None
    if not data:
        return None

    if tool_name == "classify_intent":
        return f"Intent identified as '{data.get('intent')}'"
    if tool_name == "get_credit_risk":
        band = data.get("risk_band")
        prob = data.get("default_probability")
        return f"Risk band: {band} ({prob} default probability)" if band else None
    if tool_name == "retrieve_policy":
        n = len(data.get("results", []))
        return f"Found {n} relevant polic{'y' if n == 1 else 'ies'}"
    if tool_name == "analyze_document":
        return f"Document classified as '{data.get('doc_type')}'"
    return None


def build_workflow(tool_call_log):
    """Returns a list of {step, status, detail} describing what the system
    did, in the order it did it, plus a final synthesis step. status is
    'success' or 'error'; detail is a short plain-language summary or None."""
    steps = []
    for call in tool_call_log:
        name = call.get("tool", "unknown_step")
        result = call.get("result", {})
        ok = _is_success(result)
        steps.append({
            "step": STEP_LABELS.get(name, name),
            "status": "success" if ok else "error",
            "detail": _summarize(name, result) if ok else "Could not complete - continuing with available information",
        })

    steps.append({"step": "Final recommendation generated", "status": "success", "detail": None})
    return steps