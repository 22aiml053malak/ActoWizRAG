"""
Document repository — SQLAlchemy CRUD for the `documents` and `query_logs` tables.

This is the ONLY place in the codebase that touches these two tables directly.
Services must go through this repository, never touch the ORM models themselves.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import Document, QueryLog
from app.core.logger import get_logger

logger = get_logger(__name__)


class DocumentRepository:
    """CRUD operations for the `documents` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Create ─────────────────────────────────────────────────────────────────

    async def create(
        self,
        *,
        document_id: uuid.UUID,
        filename: str,
        file_type: str,
        storage_path: str,
    ) -> Document:
        doc = Document(
            id=document_id,
            filename=filename,
            file_type=file_type,
            storage_path=storage_path,
            status="pending",
        )
        self._session.add(doc)
        await self._session.flush()
        logger.info("Document row created", extra={"document_id": str(document_id)})
        return doc

    # ── Read ───────────────────────────────────────────────────────────────────

    async def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        result = await self._session.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def list_active(
        self, limit: int = 100, offset: int = 0
    ) -> tuple[list[Document], int]:
        """Return (items, total) for documents that are not soft-deleted."""
        q = select(Document).where(Document.deleted_at.is_(None)).order_by(
            Document.uploaded_at.desc()
        )
        count_q = select(func.count()).select_from(q.subquery())
        total = (await self._session.execute(count_q)).scalar_one()
        items = (
            await self._session.execute(q.limit(limit).offset(offset))
        ).scalars().all()
        return list(items), total

    # ── Update ─────────────────────────────────────────────────────────────────

    async def update_status(
        self,
        document_id: uuid.UUID,
        status: str,
        *,
        error_message: str | None = None,
        chunk_count: int | None = None,
    ) -> Document | None:
        doc = await self.get_by_id(document_id)
        if doc is None:
            return None
        doc.status = status
        doc.updated_at = datetime.now(tz=timezone.utc)
        if error_message is not None:
            doc.error_message = error_message
        if chunk_count is not None:
            doc.chunk_count = chunk_count
        await self._session.flush()
        logger.info(
            "Document status updated",
            extra={"document_id": str(document_id), "status": status},
        )
        return doc

    async def soft_delete(self, document_id: uuid.UUID) -> Document | None:
        """Mark document as 'deleting' and set deleted_at timestamp."""
        doc = await self.get_by_id(document_id)
        if doc is None:
            return None
        doc.status = "deleting"
        doc.deleted_at = datetime.now(tz=timezone.utc)
        doc.updated_at = datetime.now(tz=timezone.utc)
        await self._session.flush()
        logger.info("Document soft-deleted", extra={"document_id": str(document_id)})
        return doc

    async def hard_delete(self, document_id: uuid.UUID) -> bool:
        """Permanently remove the document row (called after vectors are cleaned up)."""
        doc = await self.get_by_id(document_id)
        if doc is None:
            return False
        await self._session.delete(doc)
        await self._session.flush()
        logger.info("Document hard-deleted", extra={"document_id": str(document_id)})
        return True


class QueryLogRepository:
    """CRUD operations for the `query_logs` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        query_text: str,
        filters: dict[str, Any] | None,
        top_k: int,
        result_chunk_ids: list[dict[str, Any]],
        latency_ms: int,
    ) -> QueryLog:
        log = QueryLog(
            query_text=query_text,
            filters=filters,
            top_k=top_k,
            result_chunk_ids=result_chunk_ids,
            latency_ms=latency_ms,
        )
        self._session.add(log)
        await self._session.flush()
        return log
