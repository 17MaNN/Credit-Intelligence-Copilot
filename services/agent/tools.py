"""Tool schemas (Bedrock Converse API format) + executor functions that call
the four existing services over HTTP. Each tool maps 1:1 to a service endpoint
already built in Phases 1-4 - no new business logic here, just plumbing."""
import os
import base64
import httpx

SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY", "")
RISK_MODEL_URL = os.environ.get("RISK_MODEL_URL", "http://risk-model:8000")
DOC_CV_URL = os.environ.get("DOC_CV_URL", "http://doc-cv:8000")
NLP_URL = os.environ.get("NLP_URL", "http://nlp-classifier:8000")
RAG_URL = os.environ.get("RAG_URL", "http://rag:8000")

HEADERS = {"x-api-key": SERVICE_API_KEY}
TIMEOUT = 15.0

TOOLS = [
    {
        "toolSpec": {
            "name": "get_credit_risk",
            "description": "Score a customer's default risk from financial details.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {
                    "income": {"type": "number"},
                    "debt_to_income": {"type": "number", "description": "0 to 1"},
                    "credit_history_years": {"type": "number"},
                    "num_delinquencies": {"type": "integer"},
                    "loan_amount": {"type": "number"},
                },
                "required": ["income", "debt_to_income", "credit_history_years",
                              "num_delinquencies", "loan_amount"],
            }},
        }
    },
    {
        "toolSpec": {
            "name": "classify_intent",
            "description": "Classify a customer message into an intent: "
                            "promise_to_pay, dispute, hardship, or other.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            }},
        }
    },
    {
        "toolSpec": {
            "name": "retrieve_policy",
            "description": "Retrieve relevant internal policy text for a topic "
                            "(e.g. hardship plans, disputes, contact rules).",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            }},
        }
    },
    {
        "toolSpec": {
            "name": "analyze_document",
            "description": "Classify a document type and OCR its text from a "
                            "base64-encoded image.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"image_base64": {"type": "string"}},
                "required": ["image_base64"],
            }},
        }
    },
]


def _post_json(url, payload):
    r = httpx.post(url, json=payload, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_credit_risk(income, debt_to_income, credit_history_years, num_delinquencies, loan_amount):
    return _post_json(f"{RISK_MODEL_URL}/predict", {
        "income": income, "debt_to_income": debt_to_income,
        "credit_history_years": credit_history_years,
        "num_delinquencies": num_delinquencies, "loan_amount": loan_amount,
    })


def classify_intent(text):
    return _post_json(f"{NLP_URL}/classify-text", {"text": text})


def retrieve_policy(query, top_k=3):
    return _post_json(f"{RAG_URL}/retrieve", {"query": query, "top_k": top_k})


def analyze_document(image_base64):
    raw = base64.b64decode(image_base64)
    files = {"file": ("doc.png", raw, "image/png")}
    r = httpx.post(f"{DOC_CV_URL}/analyze-document", files=files, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


EXECUTORS = {
    "get_credit_risk": get_credit_risk,
    "classify_intent": classify_intent,
    "retrieve_policy": retrieve_policy,
    "analyze_document": analyze_document,
}