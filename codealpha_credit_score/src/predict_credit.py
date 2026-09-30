import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"


def predict_creditworthiness(input_data):
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0, 1]

    result = {
        "creditworthy": int(prediction),
        "probability_of_creditworthiness": float(probability),
    }
    return result


if __name__ == "__main__":
    sample = {
        "age": 35,
        "income": 85000,
        "monthly_debt": 450,
        "debt_to_income_ratio": 0.18,
        "credit_utilization": 0.35,
        "payment_history_score": 0.92,
        "late_payments": 1,
        "savings_balance": 15000,
        "employment_years": 7,
        "loan_count": 2,
        "credit_score": 710,
    }

    print(predict_creditworthiness(sample))
