"""Wraps Bedrock's Converse API and runs the tool-calling loop. Kept separate
from main.py so the loop logic can be unit-tested without spinning up FastAPI."""
import os
import boto3
from tools import TOOLS, EXECUTORS

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0"
)
MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = (
    "You are a lending/collections operations assistant. Use the available "
    "tools to gather facts (risk score, customer intent, relevant policy) "
    "before making a recommendation. Always cite which policy applies when "
    "you recommend an action. Keep the final answer concise and structured: "
    "state the recommended action, the reasoning, and any policy reference."
)


def _client():
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def run_agent(user_message: str):
    """Runs the tool-calling loop. Returns (final_text, tool_call_log)."""
    client = _client()
    messages = [{"role": "user", "content": [{"text": user_message}]}]
    tool_call_log = []

    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.converse(
            modelId=MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig={"tools": TOOLS},
        )
        output_message = resp["output"]["message"]
        messages.append(output_message)

        if resp["stopReason"] != "tool_use":
            final_text = "".join(
                block.get("text", "") for block in output_message["content"]
            )
            return final_text, tool_call_log

        tool_results = []
        for block in output_message["content"]:
            if "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            name, tool_id, tool_input = tool_use["name"], tool_use["toolUseId"], tool_use["input"]

            try:
                result = EXECUTORS[name](**tool_input)
                status = "success"
            except Exception as e:
                result = {"error": str(e)}
                status = "error"

            tool_call_log.append({"tool": name, "input": tool_input, "result": result})
            tool_results.append({
                "toolResult": {
                    "toolUseId": tool_id,
                    "content": [{"json": result}],
                    "status": status,
                }
            })

        messages.append({"role": "user", "content": tool_results})

    return "Reached max tool-call rounds without a final answer.", tool_call_log