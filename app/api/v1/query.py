"""
Query API router — POST /api/v1/query
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

# Module-level singleton — built once when the module is first imported,
# so the same ML models are reused for every request without lru_cache
# staleness issues.
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        logger.info("Building RAGService (first request)")
        vector_repo   = VectorRepository()
        embedding_svc = EmbeddingService()
        retrieval_svc = RetrievalService(
            vector_repo=vector_repo,
            embedding_service=embedding_svc,
        )
        llm_svc = LLMService()
        _rag_service = RAGService(retrieval_service=retrieval_svc, llm_service=llm_svc)
    return _rag_service


@router.post(
    "",
    response_model=QueryResponse,
    summary="Semantic search with optional LLM answer generation",
)
async def query_documents(
    request: QueryRequest,
    session: AsyncSession = Depends(get_db_session),
) -> QueryResponse:
    start = time.time()

    rag = get_rag_service()

    loop = asyncio.get_running_loop()
    response: QueryResponse = await loop.run_in_executor(
        None,
        lambda: rag.query(
            query_text=request.query,
            top_k=request.top_k,
            generate_answer=request.generate_answer,
        ),
    )

    latency_ms = int((time.time() - start) * 1000)

    result_chunk_ids: list[dict[str, Any]] = [
        {"node_id": r.node_id, "score": r.score} for r in response.results
    ]
    asyncio.create_task(
        _log_query(
            query_text=request.query,
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
        },
    )
    return response


async def _log_query(
    query_text: str,
    top_k: int,
    result_chunk_ids: list[dict],
    latency_ms: int,
) -> None:
    try:
        from app.models.orm import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            log_repo = QueryLogRepository(session)
            await log_repo.create(
                query_text=query_text,
                filters=None,
                top_k=top_k,
                result_chunk_ids=result_chunk_ids,
                latency_ms=latency_ms,
            )
            await session.commit()
    except Exception as exc:
        logger.warning("Failed to write query log", extra={"error": str(exc)})
