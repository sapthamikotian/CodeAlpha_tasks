# Credit Scoring Model

This project builds a machine learning credit scoring model to predict whether an individual is creditworthy based on financial history and behavioral indicators.

## Objective

The goal is to classify applicants as either:

- `1` = creditworthy
- `0` = not creditworthy

The model uses features such as income, debt, payment history, credit utilization, and employment stability.

## Included algorithms

- Logistic Regression
- Decision Tree
- Random Forest

## Model evaluation metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Project structure

- `src/generate_credit_data.py` – synthetic data generation
- `src/train_credit_model.py` – model training, comparison, and evaluation
- `data/credit_data.csv` – generated dataset
- `models/` – saved trained model

## Quick start

1. Install dependencies:
   ```bash
   py -m pip install -r requirements.txt
   ```

2. Generate the dataset:
   ```bash
   py src/generate_credit_data.py
   ```

3. Train and compare models:
   ```bash
   py src/train_credit_model.py
   ```

4. Review the printed metrics and the saved model in the `models` folder.

## Example features used

- income
- debt_to_income_ratio
- monthly_debt
- credit_utilization
- payment_history_score
- late_payments
- savings_balance
- employment_years
- loan_count

## Notes

This project uses a synthetic dataset so it can run immediately in a local environment without needing external banking data.
