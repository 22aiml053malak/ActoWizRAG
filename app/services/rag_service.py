"""
RAG service — coordinates retrieval_service + llm_service for /query endpoint.

This service is the single point of truth for the full query pipeline:
  retrieve → (optional) generate_answer → log
"""

from __future__ import annotations

import time
from typing import Any

from app.core.exceptions import LLMProviderNotConfiguredError
from app.core.logger import get_logger
from app.models.response import ChunkResult, QueryResponse
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


class RAGService:
    """
    Top-level service called by the /query API route.

    It:
      1. Delegates retrieval to RetrievalService (retrieve→rerank→window-replace).
      2. Optionally synthesises an answer via LLMService.
      3. Returns a fully-populated QueryResponse.

    Query logging is handled at the API layer (documents.py / query.py)
    so that it can be done asynchronously without blocking this service.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._llm_service = llm_service

    def query(
        self,
        *,
        query_text: str,
        top_k: int = 5,
        document_id: str | None = None,
        filters: dict[str, Any] | None = None,
        generate_answer: bool = False,
    ) -> QueryResponse:
        """
        Run the full RAG query pipeline.

        Args:
            query_text:      Natural language query.
            top_k:           Number of chunks to return.
            document_id:     Optional document UUID filter.
            filters:         Optional metadata key-value filters.
            generate_answer: If True, call LLM Gateway to synthesize an answer.

        Returns:
            QueryResponse ready for serialisation.
        """
        start_ms = time.time()

        # ── Retrieval ──────────────────────────────────────────────────────────
        results: list[ChunkResult] = self._retrieval_service.retrieve(
            query_text,
            top_k=top_k,
            document_id=document_id,
            filters=filters,
        )

        logger.info(
            "Retrieval complete",
            extra={"query": query_text[:80], "results": len(results)},
        )

        # ── Answer generation (optional) ───────────────────────────────────────
        answer: str | None = None
        sources: list[str] | None = None

        if generate_answer:
            if not results:
                answer = "No relevant documents were found to answer the question."
                sources = []
            else:
                try:
                    answer, sources = self._llm_service.generate_answer(
                        query=query_text,
                        chunks=results,
                    )
                except LLMProviderNotConfiguredError as exc:
                    # Graceful degradation: return chunks but include the error as answer.
                    answer = str(exc)
                    sources = []
                    logger.warning("LLM provider not configured; returning chunks only")

        latency_ms = int((time.time() - start_ms) * 1000)

        return QueryResponse(
            query=query_text,
            top_k=top_k,
            results=results,
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
        )
