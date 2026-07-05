"""
Synchronous ingestion helper — used when Celery/Redis is unavailable
and as the implementation behind the Celery task.
"""

from __future__ import annotations

import traceback
import uuid

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def _update_status(
    session,
    document_id: str,
    status: str,
    *,
    error_message: str | None = None,
    chunk_count: int | None = None,
) -> None:
    from datetime import datetime, timezone

    updates: dict = {"status": status, "updated_at": datetime.now(tz=timezone.utc)}
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


def run_ingestion(document_id: str) -> int:
    """
    Run load → chunk → embed → store for one document.

    Returns chunk count on success. Updates document status in DB.
    """
    engine = create_engine(settings.DATABASE_SYNC_URL, pool_pre_ping=True)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    session = Session()

    try:
        _update_status(session, document_id, "processing")

        row = session.execute(
            text(
                "SELECT filename, file_type, storage_path "
                "FROM documents WHERE id = :doc_id"
            ),
            {"doc_id": uuid.UUID(document_id)},
        ).fetchone()

        if row is None:
            raise ValueError(f"Document {document_id} not found in DB")

        from app.repositories.vector_repository import VectorRepository
        from app.services.chunking_service import ChunkingService
        from app.services.embedding_service import EmbeddingService
        from app.services.ingestion_service import IngestionService

        ingestion_svc = IngestionService(
            vector_repo=VectorRepository(),
            chunking_service=ChunkingService(),
            embedding_service=EmbeddingService(),
        )

        chunk_count = ingestion_svc.ingest(
            document_id=document_id,
            storage_path=row.storage_path,
            filename=row.filename,
            file_type=row.file_type,
        )

        _update_status(session, document_id, "completed", chunk_count=chunk_count)
        logger.info(
            "Ingestion completed",
            extra={"document_id": document_id, "chunk_count": chunk_count},
        )
        return chunk_count

    except Exception as exc:
        tb = traceback.format_exc()
        error_detail = getattr(exc, "detail", str(exc))
        logger.error(
            "Ingestion failed",
            extra={"document_id": document_id, "error": error_detail, "traceback": tb},
        )
        try:
            _update_status(
                session,
                document_id,
                "failed",
                error_message=f"{type(exc).__name__}: {error_detail}",
            )
        except Exception:
            pass
        raise
    finally:
        session.close()
        engine.dispose()
