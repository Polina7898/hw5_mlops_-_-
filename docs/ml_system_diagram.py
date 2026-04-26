"""
Схема ML-системы для размытия лиц на видео
Запуск: python docs/ml_system_diagram.py
Артефакт: docs/ml_system_face_blur.png
"""
from diagrams import Cluster, Diagram, Edge
from diagrams.onprem.client import Users
from diagrams.onprem.compute import Server
from diagrams.onprem.queue import Kafka
from diagrams.onprem.inmemory import Redis
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.monitoring import Grafana, Prometheus
from diagrams.onprem.mlops import Mlflow
from diagrams.onprem.workflow import Airflow
from diagrams.programming.framework import Fastapi
from diagrams.aws.storage import S3


graph_attr = {
    "fontsize": "16",
    "bgcolor": "white",
    "splines": "spline",
}


with Diagram(
    "ML-система размытия лиц на видео",
    filename="docs/ml_system_face_blur",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
):
    user = Users("Загрузка видео")

    with Cluster("Ingest"):
        api = Fastapi("Upload API")
        raw_storage = S3("S3: raw video")
        broker = Kafka("Очередь кадров")

    with Cluster("Препроцессинг"):
        splitter = Server("Frame splitter\n(GPU pool)")
        cache = Redis("Cache: dedup")

    with Cluster("Inference workers (parallel)"):
        det1 = Server("Detector #1\n(YOLO/RetinaFace)")
        det2 = Server("Detector #2")
        det3 = Server("Detector #3")

    with Cluster("Постпроцессинг"):
        blur = Server("Mosaic / Gaussian\nblur worker")
        encoder = Server("Video encoder\n(ffmpeg)")
        result_storage = S3("S3: blurred video")

    with Cluster("MLOps контур"):
        registry = Mlflow("MLflow Registry")
        airflow = Airflow("Retraining DAG")
        feast_db = PostgreSQL("Feature Store\n(Postgres)")

    with Cluster("Observability"):
        prom = Prometheus("Метрики latency / FPS")
        graf = Grafana("Дашборды и SLA")

    user >> api >> raw_storage >> broker
    broker >> splitter >> cache
    cache >> Edge(label="batch") >> [det1, det2, det3]
    [det1, det2, det3] >> blur >> encoder >> result_storage
    user << Edge(label="ссылка на результат") << result_storage

    registry >> Edge(style="dashed", label="serve model") >> [det1, det2, det3]
    airflow >> Edge(style="dashed") >> registry
    feast_db >> Edge(style="dashed", label="features") >> [det1, det2, det3]

    [det1, det2, det3, blur, encoder] >> Edge(style="dotted") >> prom
    prom >> graf
