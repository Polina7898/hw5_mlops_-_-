"""
Стадия prepare пайплайна DVC

Загружает сырой датасет Iris, делает базовую очистку,
кодирует целевую переменную и разбивает выборку на train/test
"""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path("data/raw/iris.csv")
PROCESSED_DIR = Path("data/processed")


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    params = load_params()["prepare"]

    df = pd.read_csv(RAW_DATA_PATH)

    # базовая очистка: убираем пустые строки и дубликаты
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # кодируем целевой класс в числовой формат
    target_col = params.get("target", "variety")
    df[target_col] = df[target_col].astype("category").cat.codes

    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df[target_col],
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"[prepare] train: {train_df.shape}, test: {test_df.shape}")


if __name__ == "__main__":
    main()
