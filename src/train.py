"""
Стадия train пайплайна DVC

Обучает модель на подготовленных данных, считает метрики
и логирует параметры/метрики/артефакты в локальный MLflow
"""
import json
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("metrics.json")


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(params: dict):
    name = params["train"]["model"]
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["train"]["n_estimators"],
            max_depth=params["train"]["max_depth"],
            random_state=params["train"]["random_state"],
        )
    if name == "logistic_regression":
        return LogisticRegression(
            C=params["train"].get("C", 1.0),
            max_iter=params["train"].get("max_iter", 200),
            random_state=params["train"]["random_state"],
        )
    raise ValueError(f"Неизвестная модель: {name}")


def main() -> None:
    params = load_params()
    target_col = params["prepare"]["target"]

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    x_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    x_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=params["mlflow"]["run_name"]):
        model = build_model(params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro"),
            "recall": recall_score(y_test, y_pred, average="macro"),
            "f1": f1_score(y_test, y_pred, average="macro"),
        }

        mlflow.log_params(params["train"])
        mlflow.log_param("test_size", params["prepare"]["test_size"])
        mlflow.log_metrics(metrics)

        signature = infer_signature(x_test, y_pred)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=x_test.head(3),
        )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(str(MODEL_PATH))

        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"[train] {metrics}")


if __name__ == "__main__":
    main()
