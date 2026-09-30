import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def generate_credit_data(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 70, size=n_samples)
    income = rng.lognormal(mean=10.8, sigma=0.45, size=n_samples)
    income = np.clip(income, 20000, 220000).astype(float)

    monthly_debt = rng.uniform(200, 4500, size=n_samples)
    debt_to_income_ratio = rng.uniform(0.05, 0.9, size=n_samples)
    credit_utilization = rng.uniform(0.10, 0.95, size=n_samples)
    payment_history_score = rng.uniform(0.35, 0.98, size=n_samples)
    late_payments = rng.integers(0, 12, size=n_samples)
    savings_balance = rng.uniform(0, 60000, size=n_samples)
    employment_years = rng.integers(0, 20, size=n_samples)
    loan_count = rng.integers(0, 6, size=n_samples)
    credit_score = rng.integers(400, 850, size=n_samples)

    risk_score = (
        0.00002 * income
        - 0.28 * payment_history_score
        + 0.50 * credit_utilization
        + 0.60 * debt_to_income_ratio
        + 0.35 * late_payments
        - 0.00015 * savings_balance
        - 0.25 * employment_years / 10
        - 0.15 * (loan_count / 2)
    )

    utility = (
        1.2
        + 0.000015 * income
        - 0.0004 * monthly_debt
        - 0.60 * credit_utilization
        - 0.30 * debt_to_income_ratio
        - 0.15 * late_payments
        + 0.00001 * savings_balance
        + 0.20 * payment_history_score
    )

    latent_probability = 1 / (1 + np.exp(-(utility + rng.normal(0, 0.9, n_samples))))
    creditworthy = (latent_probability > 0.48).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "monthly_debt": monthly_debt,
            "debt_to_income_ratio": debt_to_income_ratio,
            "credit_utilization": credit_utilization,
            "payment_history_score": payment_history_score,
            "late_payments": late_payments,
            "savings_balance": savings_balance,
            "employment_years": employment_years,
            "loan_count": loan_count,
            "credit_score": credit_score,
            "creditworthy": creditworthy,
        }
    )

    return df


if __name__ == "__main__":
    dataset = generate_credit_data()
    output_path = DATA_DIR / "credit_data.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Generated dataset with {len(dataset)} rows at {output_path}")
