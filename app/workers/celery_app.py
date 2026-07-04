"""
Celery application factory.

Celery is configured to use Redis as both the broker and the result backend.
Task serialisation uses JSON for portability and debuggability.
"""

from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "actowiz_rag",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.ingestion_tasks"],
)

celery_app.conf.update(
    # ── Serialisation ──────────────────────────────────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # ── Reliability ───────────────────────────────────────────────────────────
    task_acks_late=True,              # ACK only after the task completes (safe re-queue on crash)
    worker_prefetch_multiplier=1,     # One task at a time per worker process
    task_reject_on_worker_lost=True,  # Re-queue if the worker dies mid-task
    # ── Result expiry ─────────────────────────────────────────────────────────
    result_expires=3600,             # Keep task results for 1 hour
    # ── Timezone ──────────────────────────────────────────────────────────────
    timezone="UTC",
    enable_utc=True,
    # ── Retry defaults ────────────────────────────────────────────────────────
    task_max_retries=3,
    task_default_retry_delay=60,     # seconds
)
