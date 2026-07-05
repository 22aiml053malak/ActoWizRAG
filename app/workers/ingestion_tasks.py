"""
Celery ingestion tasks.

Two tasks:
  - ingest_document_task(document_id): load → chunk → embed → store
  - delete_document_task(document_id): delete vectors → hard-delete DB row

Both tasks:
  - Update document status in Postgres at every meaningful state transition.
  - Catch and log ALL exceptions with full tracebacks.
  - Never silently succeed while leaving the system in a broken state.

Partial failure handling (delete_document_task):
  If vector deletion succeeds but DB row deletion fails, the document remains
  in 'deleting' status. Re-running delete_document_task with the same document_id
  is safe — vector deletion is idempotent.
"""

from __future__ import annotations

import traceback
import uuid

from celery import Task

from app.core.logger import get_logger
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


def _make_sync_session():
    """Create a synchronous SQLAlchemy session for use inside Celery tasks."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings

    engine = create_engine(settings.DATABASE_SYNC_URL, pool_pre_ping=True)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    return Session(), engine


def _update_document_status_sync(
    session,
    document_id: str,
    status: str,
    *,
    error_message: str | None = None,
    chunk_count: int | None = None,
) -> None:
    """Synchronous status update used inside Celery tasks (no async event loop)."""
    from datetime import datetime, timezone
    from sqlalchemy import text

    updates = {"status": status, "updated_at": datetime.now(tz=timezone.utc)}
    if error_message is not None:
        updates["error_message"] = error_message
    if chunk_count is not None:
        updates["chunk_count"] = chunk_count

    set_clauses = ", ".join(f"{k} = :{k}" for k in updates)
    params = {**updates, "doc_id": uuid.UUID(document_id)}
    session.execute(
        text(f"UPDATE documents SET {set_clauses} WHERE id = :doc_id"),
        params,
    )
    session.commit()


@celery_app.task(
    bind=True,
    name="ingest_document_task",
    max_retries=3,
    default_retry_delay=30,
)
def ingest_document_task(self: Task, document_id: str) -> dict:
    """
    Background task: ingest a document from disk into the vector store.

    Delegates to run_ingestion() which handles status updates and error handling.
    """
    logger.info("Ingestion task started", extra={"document_id": document_id})
    try:
        from app.services.sync_ingest import run_ingestion

        chunk_count = run_ingestion(document_id)
        return {"document_id": document_id, "status": "completed", "chunk_count": chunk_count}
    except Exception as exc:
        tb = traceback.format_exc()
        error_detail = getattr(exc, "detail", str(exc))
        logger.error(
            "Ingestion task failed",
            extra={"document_id": document_id, "error": error_detail, "traceback": tb},
        )
        raise


@celery_app.task(
    bind=True,
    name="delete_document_task",
    max_retries=3,
    default_retry_delay=30,
)
def delete_document_task(self: Task, document_id: str) -> dict:
    """
    Background task: delete all vector-store nodes for a document, then hard-delete the DB row.

    Partial failure handling:
      - If vector deletion fails → task raises, document stays in 'deleting' status.
        Re-running the task is safe (deletion is idempotent).
      - If vector deletion succeeds but DB row deletion fails → same: stays 'deleting'.
      - If both succeed → document is fully cleaned up.
    """
    logger.info("Delete task started", extra={"document_id": document_id})

    session, engine = _make_sync_session()
    vectors_deleted = False

    try:
        # ── Step 1: Delete vectors ─────────────────────────────────────────────
        from app.repositories.vector_repository import VectorRepository
        from app.services.ingestion_service import IngestionService
        from app.services.chunking_service import ChunkingService
        from app.services.embedding_service import EmbeddingService

        vector_repo = VectorRepository()
        # IngestionService.delete() handles the two-phase deletion with fallback.
        ingestion_svc = IngestionService(
            vector_repo=vector_repo,
            chunking_service=ChunkingService(),
            embedding_service=EmbeddingService(),
        )
        ingestion_svc.delete(document_id)
        vectors_deleted = True

        # ── Step 2: Hard-delete the DB row ─────────────────────────────────────
        from sqlalchemy import text as sql_text

        session.execute(
            sql_text("DELETE FROM documents WHERE id = :doc_id"),
            {"doc_id": uuid.UUID(document_id)},
        )
        session.commit()

        logger.info(
            "Delete task completed",
            extra={"document_id": document_id, "vectors_deleted": vectors_deleted},
        )
        return {"document_id": document_id, "status": "deleted"}

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(
            "Delete task failed",
            extra={
                "document_id": document_id,
                "vectors_deleted": vectors_deleted,
                "error": str(exc),
                "traceback": tb,
            },
        )
        # Leave the document in 'deleting' status — re-running is safe.
        raise

    finally:
        session.close()
        engine.dispose()
