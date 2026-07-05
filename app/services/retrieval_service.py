"""
Retrieval service — embed → cosine search → window-expand → return chunks.

Deliberately simple:
  1. Embed the query with the same model used at ingestion time.
  2. Cosine vector search via PGVectorStore (no metadata filters).
  3. Replace each node's tight text with its stored window for richer LLM context.
  4. Map to ChunkResult response objects.

The cross-encoder reranker was removed because it was silently dropping all
results when nodes had empty text fields — cosine similarity alone is fast
and accurate enough for the current corpus size.
"""

from __future__ import annotations

from llama_index.core.schema import BaseNode, NodeWithScore

from app.core.logger import get_logger
from app.models.response import ChunkResult
from app.repositories.vector_repository import VectorRepository
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)

_INVALID_METADATA_VALUES = {"", "none", "null", "undefined", "string"}


class RetrievalService:

    def __init__(
        self,
        vector_repo: VectorRepository,
        embedding_service: EmbeddingService,
    ) -> None:
        self._vector_repo = vector_repo
        self._embedding_service = embedding_service
        logger.info("RetrievalService initialised (cosine-only, no reranker)")

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, *, top_k: int = 5) -> list[ChunkResult]:
        """
        Retrieve the top_k most similar chunks for *query*.

        Args:
            query:  Natural language question or keyword string.
            top_k:  Number of chunks to return.

        Returns:
            List of ChunkResult ordered by cosine similarity (best first).
        """
        # Step 1 — embed
        query_embedding = self._embedding_service.embed_query(query)
        logger.debug("Query embedded", extra={"query": query[:80]})

        # Step 2 — vector search (fetch 3× top_k so window expansion has room)
        fetch_k = max(top_k * 3, 15)
        candidates: list[NodeWithScore] = self._vector_repo.search(
            query_embedding=query_embedding,
            similarity_top_k=fetch_k,
        )
        logger.info(
            "Vector search complete",
            extra={"candidates": len(candidates), "fetch_k": fetch_k},
        )

        if not candidates:
            return []

        # Step 3 — hydrate node text (PGVectorStore sometimes returns empty .text)
        for nws in candidates:
            self._hydrate(nws.node)

        # Step 4 — window expansion: prefer the wider context window
        for nws in candidates:
            window = nws.node.metadata.get("window", "")
            if window and window.strip():
                nws.node.set_content(window.strip())

        # Step 5 — keep top_k, map to response schema
        candidates = candidates[:top_k]
        results: list[ChunkResult] = []
        for nws in candidates:
            node  = nws.node
            meta  = node.metadata or {}
            content = self._get_text(node)
            if not content:
                continue
            results.append(ChunkResult(
                node_id=node.node_id,
                document_id=self._first_valid(meta, "document_id", "source_document_id"),
                filename=meta.get("filename") or meta.get("file_name"),
                content=content,
                score=float(nws.score or 0.0),
                metadata=meta,
            ))

        logger.info("Retrieval complete", extra={"results": len(results)})
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_text(node: BaseNode) -> str:
        """Return usable text from a node, falling back to metadata fields."""
        try:
            t = node.get_content(metadata_mode="none").strip()
        except TypeError:
            t = node.get_content().strip()
        if t:
            return t
        meta = node.metadata or {}
        for key in ("window", "original_text"):
            v = meta.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def _hydrate(self, node: BaseNode) -> None:
        """Ensure node.text is non-empty so downstream processors work."""
        current = self._get_text(node)
        if not current:
            return
        try:
            existing = node.get_content(metadata_mode="none").strip()
        except TypeError:
            existing = node.get_content().strip()
        if existing:
            return
        if hasattr(node, "set_content"):
            node.set_content(current)
        elif hasattr(node, "text"):
            node.text = current  # type: ignore[attr-defined]

    @staticmethod
    def _first_valid(meta: dict, *keys: str) -> str | None:
        for key in keys:
            v = meta.get(key)
            if v and str(v).strip().lower() not in _INVALID_METADATA_VALUES:
                return str(v).strip()
        return None
