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
# Logical name passed to PGVectorStore.from_params(table_name=...).
CHUNK_TABLE_NAME = "document_chunks"
# PGVectorStore 0.1.x materialises the physical table as data_{table_name}.
PHYSICAL_CHUNK_TABLE_NAME = f"data_{CHUNK_TABLE_NAME}"
SOURCE_DOCUMENT_ID_KEY = "source_document_id"


def _build_pg_vector_store() -> PGVectorStore:
    """
    Construct a PGVectorStore connected to the shared Postgres instance.

    PGVectorStore.from_params uses psycopg2 (sync) under the hood;
    we pass the sync DATABASE_SYNC_URL because LlamaIndex manages its own
    connection pool separately from our async SQLAlchemy engine.
    """
    return PGVectorStore.from_params(
        connection_string=settings.DATABASE_SYNC_URL,
        async_connection_string=settings.DATABASE_URL,
        table_name=CHUNK_TABLE_NAME,
        embed_dim=settings.EMBED_DIM,    # ← must be 384; a mismatch here causes
                                          #   "different vector dimensions" errors
        hybrid_search=False,
        text_search_config="english",
        perform_setup=True,
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
        self._backfill_source_document_id_metadata()
        logger.info("Nodes added to vector store", extra={"count": len(ids)})
        return ids

    def backfill_empty_text(self) -> int:
        """
        One-time repair: copy metadata_->>'original_text' into the `text` column
        for any rows where `text` is empty/null.

        PGVectorStore silently excludes rows with empty text from vector search,
        which causes queries to return zero results even when embeddings and
        metadata are intact. Rows ingested by older pipeline versions that stored
        content only in metadata fields are affected.

        Returns the number of rows updated.
        """
        import sqlalchemy as sa

        sql = sa.text(
            f"""
            UPDATE {PHYSICAL_CHUNK_TABLE_NAME}
            SET text = COALESCE(
                NULLIF(metadata_->>'original_text', ''),
                NULLIF(metadata_->>'window', '')
            )
            WHERE (text IS NULL OR trim(text) = '')
              AND COALESCE(
                  NULLIF(metadata_->>'original_text', ''),
                  NULLIF(metadata_->>'window', '')
              ) IS NOT NULL
            """
        )
        engine = sa.create_engine(settings.DATABASE_SYNC_URL)
        try:
            with engine.connect() as conn:
                result = conn.execute(sql)
                conn.commit()
                count = result.rowcount
            logger.info(
                "Backfilled empty text column from metadata",
                extra={"rows_updated": count, "table": PHYSICAL_CHUNK_TABLE_NAME},
            )
            return count
        except Exception as exc:
            logger.error(
                "Failed to backfill empty text",
                extra={"error": str(exc)},
            )
            raise
        finally:
            engine.dispose()

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
            self._backfill_source_document_id_metadata(conn)
            conn.execute(
                sa.text(
                    f"DELETE FROM {PHYSICAL_CHUNK_TABLE_NAME} "
                    f"WHERE metadata_->>'{SOURCE_DOCUMENT_ID_KEY}' = :doc_id "
                    "OR metadata_->>'document_id' = :doc_id"
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
    ) -> list[NodeWithScore]:
        """
        Retrieve top-k chunks by cosine similarity. No metadata filtering.
        """
        query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=similarity_top_k,
            mode=VectorStoreQueryMode.DEFAULT,
        )

        result = self._store.query(query)

        nodes_with_scores: list[NodeWithScore] = []
        nodes        = result.nodes or []
        similarities = result.similarities or []

        if len(similarities) < len(nodes):
            similarities = list(similarities) + [0.0] * (len(nodes) - len(similarities))

        for node, score in zip(nodes, similarities):
            nodes_with_scores.append(NodeWithScore(node=node, score=float(score or 0.0)))

        if not nodes_with_scores:
            stored = self._count_stored_chunks()
            logger.warning(
                "Vector search returned no candidates",
                extra={"top_k": similarity_top_k, "stored_chunks": stored},
            )
        else:
            logger.debug(
                "Vector search complete",
                extra={"candidates": len(nodes_with_scores)},
            )

        return nodes_with_scores

    def _count_stored_chunks(self) -> int | None:
        """Return row count in the physical pgvector table (for diagnostics)."""
        import sqlalchemy as sa

        try:
            engine = sa.create_engine(settings.DATABASE_SYNC_URL)
            with engine.connect() as conn:
                count = conn.execute(
                    sa.text(f"SELECT COUNT(*) FROM {PHYSICAL_CHUNK_TABLE_NAME}")
                ).scalar()
            engine.dispose()
            return int(count or 0)
        except Exception as exc:
            logger.debug(
                "Could not count stored chunks",
                extra={"table": PHYSICAL_CHUNK_TABLE_NAME, "error": str(exc)},
            )
            return None

    def _backfill_source_document_id_metadata(self, conn: Any | None = None) -> None:
        """
        Make the physical PGVectorStore JSON metadata filterable by our document id.

        Older rows, and some LlamaIndex serializations, store the correct app
        document id inside `_node_content` while the top-level `document_id`
        field is `"None"`. MetadataFilters only see top-level JSON keys, so we
        copy the real id into `source_document_id` and also repair
        `document_id` where possible for easier manual debugging.
        """
        import sqlalchemy as sa

        sql = sa.text(
            f"""
            WITH resolved AS (
                SELECT
                    id,
                    COALESCE(
                        NULLIF(metadata_->>'{SOURCE_DOCUMENT_ID_KEY}', ''),
                        NULLIF(NULLIF(metadata_->>'document_id', ''), 'None'),
                        CASE
                            WHEN btrim(COALESCE(metadata_->>'_node_content', '')) LIKE '{{%'
                            THEN NULLIF(
                                ((metadata_->>'_node_content')::jsonb #>> '{{metadata,document_id}}'),
                                'None'
                            )
                        END
                    ) AS source_document_id
                FROM {PHYSICAL_CHUNK_TABLE_NAME}
            )
            UPDATE {PHYSICAL_CHUNK_TABLE_NAME} AS chunks
            SET metadata_ = jsonb_set(
                jsonb_set(
                    chunks.metadata_,
                    '{{{SOURCE_DOCUMENT_ID_KEY}}}',
                    to_jsonb(resolved.source_document_id),
                    true
                ),
                '{{document_id}}',
                to_jsonb(resolved.source_document_id),
                true
            )
            FROM resolved
            WHERE chunks.id = resolved.id
              AND resolved.source_document_id IS NOT NULL
              AND (
                  chunks.metadata_->>'{SOURCE_DOCUMENT_ID_KEY}' IS DISTINCT FROM resolved.source_document_id
                  OR chunks.metadata_->>'document_id' IS DISTINCT FROM resolved.source_document_id
              )
            """
        )

        if conn is not None:
            conn.execute(sql)
            return

        engine = sa.create_engine(settings.DATABASE_SYNC_URL)
        try:
            with engine.connect() as owned_conn:
                owned_conn.execute(sql)
                owned_conn.commit()
        except Exception as exc:
            logger.debug(
                "Could not backfill vector metadata document id",
                extra={"table": PHYSICAL_CHUNK_TABLE_NAME, "error": str(exc)},
            )
        finally:
            engine.dispose()

    def get_store(self) -> PGVectorStore:
        """Expose underlying store for use in LlamaIndex index construction."""
        return self._store
