# hw5_mlops_Кузнецова_Полина

ДЗ 5 по курсу «Развертывание ML моделей»

## 1. Цель проекта

Собрать минимальный MLOps-контур, в котором:

- сырые данные хранятся через DVC, а не в git
- весь пайплайн описан в `dvc.yaml` и запускается одной командой
- параметры, метрики и модель логируются в MLflow
- есть конфигурация Feature Store на Feast
- есть схема ML-системы для размытия лиц на видео

## 2. Как запустить

```bash
git clone https://github.com/Polina7898/hw5_mlops_Kuznetsova_Polin.git
cd hw5_mlops_Kuznetsova_Polin
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/load_raw.py
dvc repro
```

## 3. Краткое описание пайплайна

В `dvc.yaml` две стадии:

- `prepare` — `src/prepare.py` чистит iris.csv, кодирует target и делит на train/test
- `train` — `src/train.py` учит RandomForest, считает accuracy/precision/recall/f1, кладёт `model.pkl` и пишет всё в MLflow

Гиперпараметры лежат в `params.yaml`, метрики — в `metrics.json`. Сырые данные сидят в DVC и подтягиваются через `dvc pull`

## 4. Где смотреть UI MLflow

После `dvc repro` запустить:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Открыть `http://127.0.0.1:5000`, эксперимент называется `hw5_iris_reproducibility`

## Дополнительно

- `feature_repo/` — конфиг Feast (шаблон postgres) и описание feature view
- `docs/ml_system_face_blur.png` — схема ML-системы для блюра лиц
