"""
Query API router — semantic search endpoint.

POST /api/v1/query accepts a natural language query and returns semantically
relevant, reranked, context-expanded chunks (optionally with an LLM answer).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import get_logger
from app.models.orm import get_db_session
from app.models.request import QueryRequest
from app.models.response import QueryResponse
from app.repositories.document_repository import QueryLogRepository
from app.repositories.vector_repository import VectorRepository
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


def _get_rag_service() -> RAGService:
    """
    Build the RAGService dependency graph.

    In production you would use FastAPI's lifespan / dependency-injection
    to share model instances across requests. For clarity here we build fresh
    service objects per request; the heavy models (HuggingFace, cross-encoder)
    are loaded once per process by EmbeddingService / RetrievalService, but
    the service objects themselves are lightweight wrappers.
    """
    vector_repo = VectorRepository()
    embedding_svc = EmbeddingService()
    retrieval_svc = RetrievalService(
        vector_repo=vector_repo,
        embedding_service=embedding_svc,
    )
    llm_svc = LLMService()
    return RAGService(retrieval_service=retrieval_svc, llm_service=llm_svc)


@router.post(
    "",
    response_model=QueryResponse,
    summary="Semantic search with optional LLM answer generation",
    description=(
        "Embeds the query, performs wide vector search (similarity_top_k=15), "
        "reranks with a cross-encoder, expands context windows, and returns "
        "the top_k most relevant chunks. "
        "Set generate_answer=true to get an LLM-synthesised answer (requires "
        "LLM_PROVIDER to be configured)."
    ),
)
async def query_documents(
    request: QueryRequest,
    session: AsyncSession = Depends(get_db_session),
) -> QueryResponse:
    start = time.time()

    rag_service = _get_rag_service()

    # Run the synchronous retrieval in a thread pool to avoid blocking the event loop.
    loop = asyncio.get_running_loop()
    response: QueryResponse = await loop.run_in_executor(
        None,
        lambda: rag_service.query(
            query_text=request.query,
            top_k=request.top_k,
            document_id=request.document_id,
            filters=request.filters,
            generate_answer=request.generate_answer,
        ),
    )

    latency_ms = int((time.time() - start) * 1000)

    # ── Async query logging (fire-and-forget, don't block the response) ────────
    result_chunk_ids: list[dict[str, Any]] = [
        {"node_id": r.node_id, "score": r.score}
        for r in response.results
    ]
    asyncio.create_task(
        _log_query(
            session=session,
            query_text=request.query,
            filters=request.filters,
            top_k=request.top_k,
            result_chunk_ids=result_chunk_ids,
            latency_ms=latency_ms,
        )
    )

    logger.info(
        "Query handled",
        extra={
            "query": request.query[:80],
            "results": len(response.results),
            "latency_ms": latency_ms,
            "generate_answer": request.generate_answer,
        },
    )
    return response


async def _log_query(
    session: AsyncSession,
    query_text: str,
    filters: dict | None,
    top_k: int,
    result_chunk_ids: list[dict],
    latency_ms: int,
) -> None:
    """Fire-and-forget coroutine to persist query logs without blocking the response."""
    try:
        log_repo = QueryLogRepository(session)
        await log_repo.create(
            query_text=query_text,
            filters=filters,
            top_k=top_k,
            result_chunk_ids=result_chunk_ids,
            latency_ms=latency_ms,
        )
        await session.commit()
    except Exception as exc:
        logger.warning("Failed to write query log", extra={"error": str(exc)})
