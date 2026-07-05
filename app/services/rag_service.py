"""
RAG service — retrieval + answer generation for the /query endpoint.
"""

from __future__ import annotations

import time

from app.core.exceptions import LLMProviderNotConfiguredError
from app.core.logger import get_logger
from app.models.response import ChunkResult, QueryResponse
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


class RAGService:

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
    ) -> None:
        self._retrieval = retrieval_service
        self._llm = llm_service

    def query(
        self,
        *,
        query_text: str,
        top_k: int = 5,
        generate_answer: bool = True,
    ) -> QueryResponse:
        start = time.time()

        results: list[ChunkResult] = self._retrieval.retrieve(
            query_text, top_k=top_k
        )
        logger.info("Retrieval done", extra={"query": query_text[:80], "results": len(results)})

        answer: str | None = None
        sources: list[str] | None = None

        if generate_answer:
            if not results:
                answer = "No relevant documents were found to answer the question."
                sources = []
            else:
                try:
                    answer, sources = self._llm.generate_answer(
                        query=query_text,
                        chunks=results,
                    )
                except LLMProviderNotConfiguredError as exc:
                    answer = str(exc)
                    sources = []

        return QueryResponse(
            query=query_text,
            top_k=top_k,
            results=results,
            answer=answer,
            sources=sources,
            latency_ms=int((time.time() - start) * 1000),
        )
