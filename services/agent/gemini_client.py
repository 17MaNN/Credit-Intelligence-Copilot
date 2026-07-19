"""Wraps the Gemini API (google-genai SDK) and runs the tool-calling loop.
Kept separate from main.py so the loop logic can be unit-tested without
spinning up FastAPI. Uses Gemini's free tier - no billing/credit card
required. Note: this uses `google-genai`, the current supported SDK -
not the deprecated `google-generativeai` package."""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tools import TOOLS, EXECUTORS

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
MAX_TOOL_ROUNDS = 6
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
MAX_PARALLEL_TOOL_CALLS = 4  # matches the number of distinct tools available

SYSTEM_PROMPT = (
    "You are a lending/collections operations assistant. Use the available "
    "tools to gather facts (risk score, customer intent, relevant policy) "
    "before making a recommendation. Always cite which policy applies when "
    "you recommend an action. Keep the final answer concise and structured: "
    "state the recommended action, the reasoning, and any policy reference. "
    "If the customer's message indicates a document was attached, call "
    "analyze_document to read it before responding - it takes no arguments."
)


def _client():
    return genai.Client(api_key=GEMINI_API_KEY)


def _send_with_retry(chat, content):
    """Retries transient 5xx errors from Gemini (e.g. temporary overload)
    with short backoff. Client errors (4xx, bad input) are not retried -
    they won't succeed on retry and should surface immediately."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return chat.send_message(content)
        except genai_errors.ServerError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise last_error


def _build_executors(image_base64):
    """Returns a fresh executor mapping for this single run_agent() call.
    analyze_document is bound to THIS call's uploaded image via closure -
    never shared/global state - so concurrent requests (Phase 13's
    same-turn parallel tool calls, or simply two different users hitting
    the service at once) can never see each other's uploaded image."""
    executors = dict(EXECUTORS)

    if image_base64:
        executors["analyze_document"] = lambda: EXECUTORS["analyze_document"](image_base64)
    else:
        executors["analyze_document"] = lambda: {"error": "no document was attached to this message"}

    return executors


def _execute_tool_call(fc, executors):
    """Runs one tool call, catching exceptions so a single failure doesn't
    take down the whole batch. Returns (name, input, result)."""
    name = fc.name
    tool_input = dict(fc.args)
    try:
        result = executors[name](**tool_input)
    except Exception as e:
        result = {"error": str(e)}
    return name, tool_input, result


def run_agent(user_message: str, image_base64: str = None):
    """Runs the tool-calling loop. Returns (final_text, tool_call_log).
    image_base64, if provided, is made available to the analyze_document
    tool for this call only - see _build_executors."""
    executors = _build_executors(image_base64)

    if image_base64:
        user_message = (
            "[A document has been attached to this message.]\n\n" + user_message
        )

    client = _client()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=TOOLS)],
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    chat = client.chats.create(model=MODEL_NAME, config=config)
    tool_call_log = []

    response = _send_with_retry(chat, user_message)

    for _ in range(MAX_TOOL_ROUNDS):
        parts = response.candidates[0].content.parts
        function_calls = [p.function_call for p in parts if p.function_call]

        if not function_calls:
            final_text = "".join(p.text for p in parts if p.text)
            return final_text, tool_call_log

        # If the model requested multiple independent tool calls in this
        # same turn, run them concurrently instead of one at a time.
        # ThreadPoolExecutor.map preserves input order in its results, so
        # the audit trail still reflects the order the model asked for
        # them in, even though they may finish in a different order.
        with ThreadPoolExecutor(max_workers=min(len(function_calls), MAX_PARALLEL_TOOL_CALLS)) as tp:
            results = list(tp.map(lambda fc: _execute_tool_call(fc, executors), function_calls))

        function_response_parts = []
        for name, tool_input, result in results:
            tool_call_log.append({"tool": name, "input": tool_input, "result": result})
            function_response_parts.append(
                types.Part.from_function_response(name=name, response={"result": result})
            )

        response = _send_with_retry(chat, function_response_parts)

    return "Reached max tool-call rounds without a final answer.", tool_call_log