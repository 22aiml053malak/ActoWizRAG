"""
Retrieval service — retrieve → rerank → window-replace.

This service implements the three-stage retrieval pipeline:

  1. Vector search (wide candidate set, similarity_top_k=15 by default)
  2. Cross-encoder reranking (cuts down to top_k, ranked by semantic relevance)
  3. Window replacement (swaps each node's tight sentence for its surrounding window)

The output is a list of ChunkResult objects ready to be serialised and returned
to the caller or passed to LLMService for answer generation.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.postprocessor import SentenceTransformerRerank, MetadataReplacementPostProcessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from app.core.config import settings
from app.core.logger import get_logger
from app.models.response import ChunkResult
from app.repositories.vector_repository import VectorRepository
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)


class RetrievalService:
    """
    Orchestrates the three-stage RAG retrieval pipeline.

    Dependencies are injected so this class can be unit-tested with mocks.
    """

    def __init__(
        self,
        vector_repo: VectorRepository,
        embedding_service: EmbeddingService,
    ) -> None:
        self._vector_repo = vector_repo
        self._embedding_service = embedding_service

        logger.info(
            "Loading reranker model",
            extra={"model": settings.RERANK_MODEL_NAME},
        )
        self._reranker = SentenceTransformerRerank(
            model=settings.RERANK_MODEL_NAME,
            top_n=settings.RERANK_TOP_N,
        )
        self._window_replacer = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )
        logger.info("RetrievalService initialised")

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """
        Run the full retrieve → rerank → window-replace pipeline.

        Args:
            query:       Natural language query string.
            top_k:       Number of chunks to return after reranking.
            document_id: Optional filter — restrict to a single document.
            filters:     Optional additional metadata filters.

        Returns:
            Ordered list of ChunkResult (best match first).
        """
        # ── Step 1: Embed the query ────────────────────────────────────────────
        query_embedding = self._embedding_service.embed_query(query)
        logger.debug("Query embedded", extra={"query": query[:80]})

        # ── Step 2: Wide vector search ─────────────────────────────────────────
        candidates: list[NodeWithScore] = self._vector_repo.search(
            query_embedding=query_embedding,
            similarity_top_k=settings.SIMILARITY_TOP_K,
            document_id=document_id,
            extra_filters=filters,
        )
        logger.info(
            "Vector search returned candidates",
            extra={"candidates": len(candidates)},
        )

        if not candidates:
            return []

        # ── Step 3: Rerank ─────────────────────────────────────────────────────
        # Override the reranker's top_n with the caller's top_k.
        self._reranker.top_n = top_k
        query_bundle = QueryBundle(query_str=query)
        reranked: list[NodeWithScore] = self._reranker.postprocess_nodes(
            candidates, query_bundle=query_bundle
        )
        logger.info(
            "Reranking complete",
            extra={"before": len(candidates), "after": len(reranked)},
        )

        # ── Step 4: Window replacement ────────────────────────────────────────
        # Swaps each node's tight sentence / code chunk text for the stored
        # surrounding window, giving the LLM richer context.
        windowed: list[NodeWithScore] = self._window_replacer.postprocess_nodes(
            reranked, query_bundle=query_bundle
        )

        # ── Step 5: Map to response schema ────────────────────────────────────
        results: list[ChunkResult] = []
        for node_with_score in windowed:
            node = node_with_score.node
            meta = node.metadata or {}
            results.append(
                ChunkResult(
                    node_id=node.node_id,
                    document_id=meta.get("document_id"),
                    filename=meta.get("filename") or meta.get("file_name"),
                    content=node.get_content(),
                    score=node_with_score.score or 0.0,
                    metadata=meta,
                )
            )

        return results
