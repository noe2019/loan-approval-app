"""Simulated loan applications (SYL Bank use case)."""
import numpy as np
import pandas as pd

PURPOSES = ["Home improvement", "Debt consolidation", "Car", "Education", "Business", "Personal"]
HOME = ["Rent", "Mortgage", "Own"]


def simulate(n=6000, seed=11):
    r = np.random.default_rng(seed)
    income = r.lognormal(11.1, 0.45, n).round(-2)
    amount = (income * r.uniform(0.05, 0.6, n)).round(-2)
    df = pd.DataFrame({
        "annual_income": income, "loan_amount": amount,
        "term_months": r.choice([12, 24, 36, 60], n, p=[.1, .2, .45, .25]),
        "credit_score": r.normal(680, 70, n).clip(300, 850).round(),
        "employment_years": r.gamma(2, 3, n).clip(0, 40).round(),
        "debt_to_income": r.beta(2, 5, n).round(3) * 60,
        "previous_defaults": r.poisson(0.25, n).clip(0, 4),
        "home_ownership": r.choice(HOME, n, p=[.45, .4, .15]),
        "purpose": r.choice(PURPOSES, n),
    })
    logit = (3.2 + 0.012 * (df.credit_score - 680) - 2.8 * df.loan_amount / df.annual_income
             - 0.06 * df.debt_to_income - 1.1 * df.previous_defaults + 0.05 * df.employment_years
             + 0.4 * (df.home_ownership != "Rent") - 0.3 * (df.purpose == "Business") - 0.006 * df.term_months)
    df["approved"] = (r.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df
