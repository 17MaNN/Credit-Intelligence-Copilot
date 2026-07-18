"""Golden evaluation set for the agent orchestrator. Each case defines the
minimum tools the agent should use (it may use more, agents have some
autonomy in ordering/strategy) and keywords the final answer should mention.
Not asserting exact wording since LLM output is non-deterministic - checking
for correct tool usage and correct grounding instead."""

CASES = [
    {
        "id": "hardship_with_financials",
        "message": (
            "Customer says they lost their job and cannot pay this month, "
            "income 40000, wants a 15000 loan, debt to income 0.5, "
            "3 delinquencies, 2 years credit history."
        ),
        "required_tools": {"classify_intent", "get_credit_risk", "retrieve_policy"},
        "response_must_contain": ["hardship"],
    },
    {
        "id": "dispute_simple",
        "message": "This charge is not mine, I never signed up for this service.",
        "required_tools": {"classify_intent", "retrieve_policy"},
        "response_must_contain": ["dispute"],
    },
    {
        "id": "promise_to_pay_simple",
        "message": "I can pay the full amount by Friday, I promise.",
        "required_tools": {"classify_intent"},
        "response_must_contain": [],
    },
    {
        "id": "neutral_inquiry",
        "message": "What are your office hours?",
        "required_tools": set(),
        "response_must_contain": [],
    },
    {
        "id": "escalation_signal",
        "message": "I can't afford basic necessities anymore and don't know what to do.",
        "required_tools": {"classify_intent", "retrieve_policy"},
        "response_must_contain": [
            ("escalat", "supervisor", "immediate attention", "urgent", "manager")
        ],
    },
]