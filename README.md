# hw5_mlops_Кузнецова_Полина

Минимальный, но полноценный MLOps-контур для классификации Iris: воспроизводимый эксперимент через Git + DVC + MLflow и описание Feature Store на базе Feast

## 1. Цель проекта

Цель — собрать рабочий каркас, в котором каждый ML-эксперимент можно повторить с нуля по одной команде, а данные, модели и метрики хранятся версионно

Что входит в контур:

- Версионирование данных и артефактов через **DVC**
- Описание ML-пайплайна (`prepare → train`) в `dvc.yaml`
- Локальный трекинг параметров, метрик и моделей в **MLflow**
- Конфигурация **Feature Store** (Feast, шаблон `postgres`)
- Схема промышленной ML-системы для размытия лиц на видео

## 2. Как запустить

```bash
git clone https://github.com/<user>/hw5_mlops_Кузнецова_Полина.git
cd hw5_mlops_Кузнецова_Полина
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/load_raw.py            # один раз: положить сырой датасет в data/raw
dvc repro                         # запустить весь пайплайн
mlflow ui --backend-store-uri sqlite:///mlflow.db   # открыть UI MLflow
```

## 3. Краткое описание пайплайна

Пайплайн описан в `dvc.yaml` и состоит из двух стадий:

- **prepare** — `src/prepare.py` читает `data/raw/iris.csv`, удаляет дубликаты и пропуски, кодирует целевую переменную и делает stratified split по `params.yaml`, складывая результат в `data/processed/{train,test}.csv`
- **train** — `src/train.py` обучает модель (по умолчанию `RandomForestClassifier`, параметры в `params.yaml`), считает accuracy / precision / recall / f1, кладёт `model.pkl` в `models/`, пишет `metrics.json` и логирует параметры, метрики, сигнатуру и модель в MLflow

Гиперпараметры и пути полностью вынесены в `params.yaml`, чтобы `dvc repro` отслеживал любые изменения и пересобирал нужные стадии. Сырые и обработанные данные не лежат в Git — на них есть `.dvc`-файлы и они приезжают через `dvc pull`

## 4. Где смотреть UI MLflow

После `dvc repro` запустите:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

UI откроется на `http://127.0.0.1:5000`, эксперимент называется `hw5_iris_reproducibility` (имя задаётся в `params.yaml → mlflow.experiment_name`). Внутри одного запуска видны:

- параметры (`model`, `n_estimators`, `max_depth`, `random_state`, `test_size`)
- метрики (`accuracy`, `precision`, `recall`, `f1`)
- артефакт `model.pkl` и сериализованная sklearn-модель с input signature

## 5. Структура репозитория

```
hw5_mlops_Кузнецова_Полина/
├── data/
│   ├── raw/            # сырые данные, через DVC
│   └── processed/      # train/test после prepare
├── src/
│   ├── load_raw.py     # одноразовая загрузка датасета
│   ├── prepare.py      # стадия prepare
│   └── train.py        # стадия train + MLflow logging
├── feature_repo/
│   ├── feature_store.yaml
│   └── example_repo.py
├── docs/
│   ├── ml_system_diagram.py
│   ├── ml_system.dot
│   └── ml_system_face_blur.png
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

## 6. Feature Store

В `feature_repo/feature_store.yaml` описан Feast с шаблоном `postgres`: онлайн и оффлайн store смотрят в один и тот же Postgres, registry — локальный `registry.db`. В `example_repo.py` объявлена сущность `flower_id` и feature view `iris_features_view` с признаками цветков

```bash
cd feature_repo
feast apply
feast ui --host 0.0.0.0 --port 8889
```

## 7. Схема ML-системы (размытие лиц)

Схема собрана в `docs/ml_system_diagram.py` (библиотека `diagrams`) и продублирована в чистом Graphviz (`docs/ml_system.dot`). Готовый PNG — `docs/ml_system_face_blur.png`

Ключевые принципы архитектуры:

- асинхронный приём видео через FastAPI и складывание в S3
- декомпозиция на кадры и пушинг батчей в Kafka
- параллельный inference на пуле GPU-воркеров с детектором лиц (RetinaFace / YOLO)
- постпроцессинг — мозаика или гауссов блюр, сборка обратно через ffmpeg
- модели берутся из MLflow Registry, признаки — из Feature Store, ретрейн оркестрируется Airflow
- наблюдаемость через Prometheus + Grafana + Alertmanager

## 8. Воспроизводимость

Эксперимент полностью воспроизводится по цепочке:

```bash
git clone <repo>
cd <repo>
pip install -r requirements.txt
dvc pull
dvc repro
```

Зафиксированы `random_state` в split и в модели, версии библиотек в `requirements.txt`, хеши данных в `.dvc`-файлах, метрики в `metrics.json` и в MLflow
