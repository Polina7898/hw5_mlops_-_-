# Feature Repo

Этот репозиторий описывает Feature Store на базе Feast с шаблоном `postgres`

## Состав

- `feature_store.yaml` — конфигурация offline и online store
- `example_repo.py` — описание сущности `flower_id` и feature view `iris_features_view`

## Запуск локально

```bash
cd feature_repo
feast apply
feast ui --host 0.0.0.0 --port 8889
```
