"""Generates a synthetic credit-risk dataset (5 features -> default probability).
Deterministic seed so training is reproducible in CI. No real financial data used."""
import numpy as np

FEATURES = ["income", "debt_to_income", "credit_history_years", "num_delinquencies", "loan_amount"]


def generate(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    income = rng.normal(60000, 20000, n).clip(15000, 200000)
    dti = rng.uniform(0, 0.6, n)
    history = rng.uniform(0, 25, n)
    delinquencies = rng.poisson(0.5, n)
    loan_amount = rng.normal(15000, 8000, n).clip(1000, 50000)

    risk_score = (
        0.4 * dti
        + 0.15 * (delinquencies / 5)
        - 0.2 * (history / 25)
        + 0.15 * (loan_amount / income)
        + rng.normal(0, 0.05, n)
    )
    default = (risk_score > np.median(risk_score)).astype(np.float32)

    X = np.stack([income, dti, history, delinquencies, loan_amount], axis=1).astype(np.float32)
    return X, default