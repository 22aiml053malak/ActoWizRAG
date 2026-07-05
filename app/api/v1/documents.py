"""
Documents API router — handles upload, list, get status, and delete.

Route handlers:
  - Validate input (via FastAPI/Pydantic).
  - Call the appropriate service.
  - Return a response.

No business logic, no raw DB/vector calls here.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, File, UploadFile, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DocumentNotFoundError, DocumentAlreadyDeletingError
from app.core.logger import get_logger
from app.models.orm import get_db_session
from app.models.response import (
    DeleteResponse,
    DocumentListResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from app.repositories.document_repository import DocumentRepository
from app.services.sync_ingest import run_ingestion
from app.utils.file_storage import get_file_type, save_upload
from app.workers.ingestion_tasks import delete_document_task, ingest_document_task

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


# ── POST /documents ────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=202,
    summary="Upload a document for ingestion",
    description=(
        "Upload a PDF, DOCX, Markdown, text, or code file. "
        "PDF and DOCX are parsed with table-aware loaders (pymupdf4llm / python-docx). "
        "The file is saved to disk and queued for async chunking + embedding. "
        "Returns 202 immediately; poll GET /documents/{id} for status."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="File to ingest"),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentUploadResponse:
    filename = file.filename or "upload"
    file_type = get_file_type(filename)  # raises UnsupportedFileTypeError if invalid
    document_id = uuid.uuid4()

    # Save file to disk first (no DB writes if this fails).
    storage_path = await save_upload(file, str(document_id))

    # Create DB row.
    repo = DocumentRepository(session)
    await repo.create(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        storage_path=storage_path,
    )

    # Commit BEFORE enqueueing Celery — otherwise the worker may not see the row yet.
    await session.commit()

    status = "pending"
    try:
        ingest_document_task.delay(str(document_id))
    except Exception as exc:
        logger.warning(
            "Celery unavailable — ingesting synchronously",
            extra={"document_id": str(document_id), "error": str(exc)},
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, run_ingestion, str(document_id))
        status = "completed"

    logger.info(
        "Document upload accepted",
        extra={"document_id": str(document_id), "file_name": filename, "status": status},
    )
    return DocumentUploadResponse(document_id=document_id, status=status)


# ── GET /documents ─────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all active documents",
)
async def list_documents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentListResponse:
    repo = DocumentRepository(session)
    items, total = await repo.list_active(limit=limit, offset=offset)
    return DocumentListResponse(
        items=[_doc_to_response(doc) for doc in items],
        total=total,
    )


# ── GET /documents/{document_id} ───────────────────────────────────────────────

@router.get(
    "/{document_id}",
    response_model=DocumentStatusResponse,
    summary="Get document ingestion status",
)
async def get_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
) -> DocumentStatusResponse:
    repo = DocumentRepository(session)
    doc = await repo.get_by_id(document_id)
    if doc is None:
        raise DocumentNotFoundError(str(document_id))
    return _doc_to_response(doc)


# ── DELETE /documents/{document_id} ───────────────────────────────────────────

@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
    summary="Delete a document and its vectors",
    description=(
        "Soft-deletes the document row immediately (sets status='deleting'), "
        "then queues a Celery task to hard-delete the vector embeddings and "
        "the database row. Returns 200 immediately."
    ),
)
async def delete_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
) -> DeleteResponse:
    repo = DocumentRepository(session)
    doc = await repo.get_by_id(document_id)

    if doc is None:
        raise DocumentNotFoundError(str(document_id))

    if doc.status == "deleting":
        raise DocumentAlreadyDeletingError(str(document_id))

    # Soft-delete immediately (status='deleting', deleted_at=now).
    await repo.soft_delete(document_id)
    await session.commit()

    try:
        delete_document_task.delay(str(document_id))
    except Exception as exc:
        logger.warning(
            "Celery unavailable — delete task not queued",
            extra={"document_id": str(document_id), "error": str(exc)},
        )

    logger.info("Delete queued", extra={"document_id": str(document_id)})
    return DeleteResponse(
        document_id=document_id,
        status="deleting",
        message="Document marked for deletion. Vectors will be removed asynchronously.",
    )


# ── Helper ─────────────────────────────────────────────────────────────────────

def _doc_to_response(doc) -> DocumentStatusResponse:
    return DocumentStatusResponse(
        document_id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        status=doc.status,
        chunk_count=doc.chunk_count or 0,
        error_message=doc.error_message,
        uploaded_at=doc.uploaded_at,
        updated_at=doc.updated_at,
        deleted_at=doc.deleted_at,
    )
