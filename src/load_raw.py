"""
Загружает сырые данные Iris из публичного источника
Запускается один раз перед dvc add data/raw/iris.csv
"""
from pathlib import Path

import pandas as pd


URL = (
    "https://gist.githubusercontent.com/netj/8836201/"
    "raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
)
RAW_PATH = Path("data/raw/iris.csv")


def main() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(URL)
    df.to_csv(RAW_PATH, index=False)
    print(f"[load_raw] сохранено {len(df)} строк в {RAW_PATH}")


if __name__ == "__main__":
    main()
