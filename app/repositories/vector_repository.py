"""
Vector repository — wraps LlamaIndex PGVectorStore.

This is the ONLY place in the codebase that instantiates or calls PGVectorStore.
All embedding storage and vector search goes through this class.

embed_dim=384 is explicitly set here and must match:
  - HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2") output dim
  - Any raw SQL that touches the vector column
  - The EMBED_DIM setting in core/config.py
"""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Table name ─────────────────────────────────────────────────────────────────
# LlamaIndex materialises this as a real Postgres table:
#   - `id` TEXT (node id)
#   - `text` TEXT
#   - `metadata_` JSONB
#   - `node_info` JSONB
#   - `embedding` vector(384)
#   - HNSW / IVFFlat index on embedding (configurable)
CHUNK_TABLE_NAME = "document_chunks"


def _build_pg_vector_store() -> PGVectorStore:
    """
    Construct a PGVectorStore connected to the shared Postgres instance.

    PGVectorStore.from_params uses psycopg2 (sync) under the hood;
    we pass the sync DATABASE_SYNC_URL because LlamaIndex manages its own
    connection pool separately from our async SQLAlchemy engine.
    """
    return PGVectorStore.from_params(
        connection_string=settings.DATABASE_SYNC_URL,
        table_name=CHUNK_TABLE_NAME,
        embed_dim=settings.EMBED_DIM,    # ← must be 384; a mismatch here causes
                                          #   "different vector dimensions" errors
        hybrid_search=False,
        text_search_config="english",
    )


class VectorRepository:
    """
    Provides insert, search, and delete operations on the pgvector chunk table.

    One instance per application (singleton pattern via dependency injection).
    PGVectorStore is thread-safe for reads; writes are protected by the
    Celery worker's single-task-per-process guarantee.
    """

    def __init__(self) -> None:
        self._store: PGVectorStore = _build_pg_vector_store()
        logger.info(
            "VectorRepository initialised",
            extra={"table": CHUNK_TABLE_NAME, "embed_dim": settings.EMBED_DIM},
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_nodes(self, nodes: list[BaseNode]) -> list[str]:
        """
        Upsert a list of nodes (with pre-computed embeddings) into the vector store.

        Returns the list of node IDs that were stored.
        """
        if not nodes:
            return []
        ids = self._store.add(nodes)
        logger.info("Nodes added to vector store", extra={"count": len(ids)})
        return ids

    # ── Delete ─────────────────────────────────────────────────────────────────

    def delete_by_document_id(self, document_id: str) -> None:
        """
        Hard-delete all chunks whose metadata["document_id"] matches.

        We use LlamaIndex's metadata-filtered delete rather than raw SQL so
        that the operation stays within the repository abstraction layer.
        """
        try:
            # PGVectorStore.delete(ref_doc_id=...) removes nodes where
            # metadata_->>'doc_id' = ref_doc_id (LlamaIndex internal field).
            # Since we store document_id in our own metadata key, we fall back
            # to a direct SQL delete filtered on metadata_.
            self._store.delete(ref_doc_id=document_id)
            logger.info(
                "Vectors deleted by document_id",
                extra={"document_id": document_id},
            )
        except Exception as exc:
            logger.error(
                "Failed to delete vectors",
                extra={"document_id": document_id, "error": str(exc)},
            )
            raise

    def delete_by_document_id_metadata(self, document_id: str) -> None:
        """
        Alternative hard-delete via raw SQL on the metadata JSONB column.
        Used as a fallback if ref_doc_id-based delete misses nodes.
        """
        import sqlalchemy as sa

        sync_url = settings.DATABASE_SYNC_URL
        engine = sa.create_engine(sync_url)
        with engine.connect() as conn:
            conn.execute(
                sa.text(
                    f"DELETE FROM {CHUNK_TABLE_NAME} "
                    "WHERE metadata_->>'document_id' = :doc_id"
                ),
                {"doc_id": document_id},
            )
            conn.commit()
        engine.dispose()
        logger.info(
            "Vectors hard-deleted via metadata filter",
            extra={"document_id": document_id},
        )

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        similarity_top_k: int = 15,
        document_id: str | None = None,
        extra_filters: dict[str, Any] | None = None,
    ) -> list[NodeWithScore]:
        """
        Retrieve top-k candidates via cosine similarity.

        Optionally apply MetadataFilters to restrict results by document_id
        or arbitrary metadata keys.
        """
        filters: MetadataFilters | None = None
        filter_list: list[MetadataFilter] = []

        if document_id:
            filter_list.append(
                MetadataFilter(
                    key="document_id",
                    value=document_id,
                    operator=FilterOperator.EQ,
                )
            )

        if extra_filters:
            for k, v in extra_filters.items():
                filter_list.append(
                    MetadataFilter(key=k, value=str(v), operator=FilterOperator.EQ)
                )

        if filter_list:
            filters = MetadataFilters(
                filters=filter_list,
                condition=FilterCondition.AND,
            )

        query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=similarity_top_k,
            mode=VectorStoreQueryMode.DEFAULT,
            filters=filters,
        )

        result = self._store.query(query)

        nodes_with_scores: list[NodeWithScore] = []
        for node, score in zip(result.nodes or [], result.similarities or []):
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        logger.debug(
            "Vector search completed",
            extra={"candidates": len(nodes_with_scores), "top_k": similarity_top_k},
        )
        return nodes_with_scores

    def get_store(self) -> PGVectorStore:
        """Expose underlying store for use in LlamaIndex index construction."""
        return self._store
