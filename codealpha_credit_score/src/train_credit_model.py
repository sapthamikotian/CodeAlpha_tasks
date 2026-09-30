from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "credit_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"

TARGET_COLUMN = "creditworthy"


def build_pipeline(model_name: str):
    numeric_features = [
        "age",
        "income",
        "monthly_debt",
        "debt_to_income_ratio",
        "credit_utilization",
        "payment_history_score",
        "late_payments",
        "savings_balance",
        "employment_years",
        "loan_count",
        "credit_score",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop",
    )

    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    elif model_name == "decision_tree":
        estimator = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Not Creditworthy", "Creditworthy"]))
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    return metrics


def main():
    dataset = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    X = dataset.drop(columns=[TARGET_COLUMN])
    y = dataset[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_names = ["logistic_regression", "decision_tree", "random_forest"]
    results = {}
    trained_models = {}

    for model_name in model_names:
        print(f"\nTraining {model_name}...")
        pipeline = build_pipeline(model_name)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)
        results[model_name] = metrics
        trained_models[model_name] = pipeline

        model_path = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path)
        print(f"Saved model to {model_path}")

    best_model_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_metrics = results[best_model_name]
    joblib.dump(trained_models[best_model_name], BEST_MODEL_PATH)

    print("\n=== Best model summary ===")
    print(f"Best model by ROC-AUC: {best_model_name}")
    print(best_metrics)
    print(f"Saved best model to {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
