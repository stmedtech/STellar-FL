from celery import Celery
from utils import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

celery_app = Celery(
    "stellar_tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["celery_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    task_send_sent_event=True,
    worker_send_task_events=True,
)

