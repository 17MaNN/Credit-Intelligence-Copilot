"""Wraps the Gemini API (google-genai SDK) and runs the tool-calling loop.
Kept separate from main.py so the loop logic can be unit-tested without
spinning up FastAPI. Uses Gemini's free tier - no billing/credit card
required. Note: this uses `google-genai`, the current supported SDK -
not the deprecated `google-generativeai` package."""
import os
import time
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tools import TOOLS, EXECUTORS

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
MAX_TOOL_ROUNDS = 6
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2

SYSTEM_PROMPT = (
    "You are a lending/collections operations assistant. Use the available "
    "tools to gather facts (risk score, customer intent, relevant policy) "
    "before making a recommendation. Always cite which policy applies when "
    "you recommend an action. Keep the final answer concise and structured: "
    "state the recommended action, the reasoning, and any policy reference."
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


def run_agent(user_message: str):
    """Runs the tool-calling loop. Returns (final_text, tool_call_log)."""
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

        function_response_parts = []
        for fc in function_calls:
            name = fc.name
            tool_input = dict(fc.args)
            try:
                result = EXECUTORS[name](**tool_input)
            except Exception as e:
                result = {"error": str(e)}

            tool_call_log.append({"tool": name, "input": tool_input, "result": result})
            function_response_parts.append(
                types.Part.from_function_response(name=name, response={"result": result})
            )

        response = _send_with_retry(chat, function_response_parts)

    return "Reached max tool-call rounds without a final answer.", tool_call_log