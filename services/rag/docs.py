"""Synthetic internal policy documents for RAG retrieval. Original text
written for this project - not copied from any real policy manual.
Swap in your actual compliance docs for production use."""

DOCS = [
    {
        "id": "hardship_policy_01",
        "text": "Customers experiencing financial hardship such as job loss, medical "
                "emergency, or income reduction may request a temporary payment plan. "
                "Hardship plans reduce the monthly payment for up to 6 months and must "
                "be approved by a supervisor before activation.",
    },
    {
        "id": "dispute_policy_01",
        "text": "If a customer disputes a charge, collections staff must pause active "
                "collection activity until the dispute is reviewed. Disputes are "
                "resolved within 10 business days. Do not report a disputed account "
                "to credit bureaus while the dispute is open.",
    },
    {
        "id": "contact_policy_01",
        "text": "Collections calls may only be made between 8am and 9pm in the "
                "customer's local time zone. Customers may request in writing that "
                "phone contact stop; all further communication must then be by mail "
                "or email only.",
    },
    {
        "id": "promise_to_pay_policy_01",
        "text": "When a customer makes a promise to pay, agents must log the promised "
                "amount and date in the account system immediately. If a promise to "
                "pay is broken, escalate to a senior collector after one missed "
                "promise; do not escalate on the first missed date automatically.",
    },
    {
        "id": "credit_reporting_policy_01",
        "text": "Accounts may only be reported as delinquent to credit bureaus after "
                "30 days of non-payment. Accounts under an active hardship plan or "
                "open dispute must not be reported as delinquent.",
    },
    {
        "id": "escalation_policy_01",
        "text": "Escalate to a supervisor immediately if a customer mentions inability "
                "to afford basic necessities, mentions self-harm, or requests legal "
                "representation. Do not continue standard collections scripting in "
                "these cases.",
    },
]