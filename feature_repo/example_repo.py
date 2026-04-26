"""
Описание сущностей и feature view для Feature Store
Источник данных - таблица iris_features в Postgres
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field, ValueType
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float32, Int64


flower = Entity(
    name="flower_id",
    join_keys=["flower_id"],
    value_type=ValueType.INT64,
    description="идентификатор цветка из датасета Iris",
)

iris_source = PostgreSQLSource(
    name="iris_source",
    query="SELECT * FROM iris_features",
    timestamp_field="event_timestamp",
)

iris_features_view = FeatureView(
    name="iris_features_view",
    entities=[flower],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="variety", dtype=Int64),
    ],
    online=True,
    source=iris_source,
    tags={"team": "mlops_hw5"},
)
