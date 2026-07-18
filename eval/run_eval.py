"""Runs the golden set against a live agent service and scores results.
Exits non-zero if the pass rate is below PASS_THRESHOLD, so this can gate
a CI deploy step. Run with: python run_eval.py
Requires the full stack (all 5 services) running and reachable."""
import os
import sys
import httpx
from golden_set import CASES

AGENT_URL = os.environ.get("AGENT_URL", "http://localhost:8000")
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY", "testkey")
PASS_THRESHOLD = float(os.environ.get("EVAL_PASS_THRESHOLD", "0.8"))
TIMEOUT = 30.0


def call_agent(message: str):
    r = httpx.post(
        f"{AGENT_URL}/handle-inquiry",
        json={"message": message},
        headers={"x-api-key": SERVICE_API_KEY},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def score_case(case: dict) -> dict:
    result = {"id": case["id"], "passed": False, "notes": []}
    try:
        resp = call_agent(case["message"])
    except Exception as e:
        result["notes"].append(f"request failed: {e}")
        return result

    if not resp.get("ok"):
        result["notes"].append(f"agent returned ok=false: {resp.get('error')}")
        return result

    data = resp["data"]
    tools_called = {t["tool"] for t in data.get("tool_calls", [])}
    missing_tools = case["required_tools"] - tools_called
    if missing_tools:
        result["notes"].append(f"missing required tools: {missing_tools}")

    failed_tool_calls = [
        t["tool"] for t in data.get("tool_calls", []) if not t["result"].get("ok", True)
    ]
    if failed_tool_calls:
        result["notes"].append(f"tool calls returned errors: {failed_tool_calls}")

    response_text = data.get("response", "").lower()
    missing_keywords = []
    for requirement in case["response_must_contain"]:
        if isinstance(requirement, (tuple, list)):
            if not any(alt.lower() in response_text for alt in requirement):
                missing_keywords.append(requirement)
        elif requirement.lower() not in response_text:
            missing_keywords.append(requirement)
    if missing_keywords:
        result["notes"].append(f"response missing expected keywords: {missing_keywords}")

    result["passed"] = not missing_tools and not failed_tool_calls and not missing_keywords
    result["tools_called"] = sorted(tools_called)
    return result


def main():
    results = [score_case(case) for case in CASES]
    passed = sum(r["passed"] for r in results)
    total = len(results)
    pass_rate = passed / total if total else 0.0

    print(f"\n{'CASE':<28} {'STATUS':<8} NOTES")
    print("-" * 70)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        notes = "; ".join(r["notes"]) if r["notes"] else f"tools: {r.get('tools_called', [])}"
        print(f"{r['id']:<28} {status:<8} {notes}")

    print("-" * 70)
    print(f"Pass rate: {passed}/{total} ({pass_rate:.0%})  threshold: {PASS_THRESHOLD:.0%}")

    if pass_rate < PASS_THRESHOLD:
        print("EVAL FAILED: pass rate below threshold")
        sys.exit(1)
    print("EVAL PASSED")


if __name__ == "__main__":
    main()