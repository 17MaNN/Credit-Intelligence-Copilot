"""Generates synthetic collections-call/email text for intent classification.
Template + slot-filling approach, deterministic seed. No real customer
transcripts used - swap in real labeled data for production."""
import random

LABELS = ["promise_to_pay", "dispute", "hardship", "other"]

TEMPLATES = {
    "promise_to_pay": [
        "I can pay the full amount by {date}.",
        "I'll send the payment on {date}, I promise.",
        "Yes I can pay {amount} next week.",
        "I plan to clear this balance by {date}.",
    ],
    "dispute": [
        "This charge is not mine, I never signed up for this.",
        "I already paid this, please check your records.",
        "This amount is wrong, I was charged twice.",
        "I want to dispute this bill, it's incorrect.",
    ],
    "hardship": [
        "I lost my job and can't pay right now.",
        "I'm going through a medical emergency, please give me more time.",
        "I can't afford this payment this month, I'm struggling financially.",
        "My income dropped and I need a payment plan.",
    ],
    "other": [
        "What is my current balance?",
        "Can you send me a copy of my statement?",
        "How do I update my mailing address?",
        "What are your office hours?",
    ],
}

DATES = ["Friday", "next Monday", "the 15th", "end of month"]
AMOUNTS = ["$200", "$450", "$1,000", "the full balance"]



def generate(n_per_class=60, seed=42):
    rng = random.Random(seed)
    texts, labels = [], []
    for idx, label in enumerate(LABELS):
        for _ in range(n_per_class):
            t = rng.choice(TEMPLATES[label])
            t = t.replace("{date}", rng.choice(DATES)).replace("{amount}", rng.choice(AMOUNTS))
            texts.append(t)
            labels.append(idx)
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)