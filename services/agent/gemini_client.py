"""Wraps the Gemini API (google-genai SDK) and runs the tool-calling loop.
Kept separate from main.py so the loop logic can be unit-tested without
spinning up FastAPI. Uses Gemini's free tier - no billing/credit card
required. Note: this uses `google-genai`, the current supported SDK -
not the deprecated `google-generativeai` package."""
import os
from google import genai
from google.genai import types
from tools import TOOLS, EXECUTORS

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = (
    "You are a lending/collections operations assistant. Use the available "
    "tools to gather facts (risk score, customer intent, relevant policy) "
    "before making a recommendation. Always cite which policy applies when "
    "you recommend an action. Keep the final answer concise and structured: "
    "state the recommended action, the reasoning, and any policy reference."
)


def _client():
    return genai.Client(api_key=GEMINI_API_KEY)


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

    response = chat.send_message(user_message)

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

        response = chat.send_message(function_response_parts)

    return "Reached max tool-call rounds without a final answer.", tool_call_log