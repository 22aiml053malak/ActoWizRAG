"""
Ingestion service — orchestrates load → chunk → embed → store.

This service is called from the Celery task (ingestion_tasks.py).
It knows about repositories but never touches SQLAlchemy or PGVectorStore
directly — it delegates through the repository layer.

document_id integrity note:
  A stored knowledge base was found with every row's top-level metadata
  showing document_id == "None" (a literal string) while the same node's
  duplicated _node_content blob had the real UUID. That means somewhere
  upstream, a falsy document_id (Python None, or an empty string) reached
  this service and got silently stringified rather than rejected. Once
  that happens, any query-time filter on document_id (e.g. "only search
  within this one document") silently matches zero rows for every
  document, because the stored value is never anything but the text
  "None". ingest() now fails loudly instead of storing bad data: it
  validates document_id up front, and re-asserts the real value on every
  node's metadata after chunking (rather than trusting that chunking_service
  set it correctly), so this class of bug can't reach the vector store from
  this code path again.
"""

from __future__ import annotations

from llama_index.core.schema import BaseNode

from app.core.exceptions import IngestionFailedError
from app.core.logger import get_logger
from app.repositories.vector_repository import VectorRepository
from app.services.chunking_service import ChunkingService
from app.services.document_loader_service import DocumentLoaderService
from app.services.embedding_service import EmbeddingService
from app.utils.file_storage import get_language

logger = get_logger(__name__)

# Values that indicate an upstream caller passed a "no id" placeholder rather
# than a real document id. Guards against exactly the bug described above:
# str(None) == "None" sneaking into metadata and silently breaking every
# document_id-scoped query from then on.
_INVALID_DOCUMENT_ID_VALUES = {"", "none", "null", "undefined", "string"}


class IngestionService:
    """
    Orchestrates the ingestion pipeline for a single document:
      1. Load raw text from disk.
      2. Chunk using ChunkingService (strategy chosen by file_type).
      3. Embed all nodes via EmbeddingService.
      4. Store nodes + embeddings in the vector store via VectorRepository.

    The Celery task handles DB status updates; this service is pure I/O.
    """

    def __init__(
        self,
        vector_repo: VectorRepository,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
        document_loader: DocumentLoaderService | None = None,
    ) -> None:
        self._vector_repo = vector_repo
        self._chunking_service = chunking_service
        self._embedding_service = embedding_service
        self._document_loader = document_loader or DocumentLoaderService()

    # ── Public API ─────────────────────────────────────────────────────────────

    def ingest(
        self,
        *,
        document_id: str,
        storage_path: str,
        filename: str,
        file_type: str,
    ) -> int:
        """
        Run the full ingestion pipeline.

        Returns:
            Number of chunks (nodes) successfully stored.

        Raises:
            IngestionFailedError on any unrecoverable failure, including a
            missing/placeholder document_id — this is checked first and
            fails fast rather than silently storing unfilterable rows.
        """
        self._validate_document_id(document_id)

        logger.info(
            "Ingestion started",
            extra={
                "document_id": document_id,
                "file_name": filename,
                "file_type": file_type,
            },
        )

        try:
            # ── 1. Load text ───────────────────────────────────────────────────
            text = self._document_loader.load(storage_path, file_type)
            if not text.strip():
                raise IngestionFailedError(document_id, "Document produced no text content")

            # ── 2. Chunk ───────────────────────────────────────────────────────
            language = get_language(filename) if file_type == "code" else None
            nodes: list[BaseNode] = self._chunking_service.chunk(
                text,
                document_id=document_id,
                file_type=file_type,
                language=language,
            )

            if not nodes:
                raise IngestionFailedError(document_id, "Chunking produced zero nodes")

            # Attach filename + re-assert document_id on every node's metadata.
            # We don't trust that chunking_service's internal assignment is
            # the last word on this field — this is the single point right
            # before storage where we guarantee both are correct, so a bug
            # anywhere upstream can't silently persist bad document_id values.
            for node in nodes:
                node.metadata["filename"] = filename
                node.metadata["document_id"] = document_id
                node.metadata["source_document_id"] = document_id

            # ── 3. Embed ───────────────────────────────────────────────────────
            nodes = self._embed_nodes(nodes)

            # ── 4. Store ───────────────────────────────────────────────────────
            stored_ids = self._vector_repo.add_nodes(nodes)
            logger.info(
                "Ingestion completed",
                extra={
                    "document_id": document_id,
                    "chunk_count": len(stored_ids),
                },
            )
            return len(stored_ids)

        except IngestionFailedError:
            raise
        except Exception as exc:
            logger.exception(
                "Unexpected ingestion error",
                extra={"document_id": document_id, "error": str(exc)},
            )
            raise IngestionFailedError(document_id, str(exc)) from exc

    def _embed_nodes(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """
        Compute and assign embeddings for all nodes in batches.
        LlamaIndex's PGVectorStore.add() requires nodes to have .embedding set.
        """
        embed_model = self._embedding_service.get_model()
        texts = [node.get_content(metadata_mode="none") for node in nodes]

        logger.info("Batch embedding nodes", extra={"count": len(texts)})
        embeddings = self._embedding_service.embed_texts(texts)

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        return nodes

    def delete(self, document_id: str) -> None:
        """
        Hard-delete all vector-store entries for a document.

        Tries the LlamaIndex ref_doc_id path first, then falls back to a
        direct metadata SQL delete to ensure no orphaned vectors remain.
        """
        self._validate_document_id(document_id)

        logger.info("Deleting vectors", extra={"document_id": document_id})
        try:
            self._vector_repo.delete_by_document_id(document_id)
        except Exception as exc:
            logger.warning(
                "Primary vector delete failed; trying fallback",
                extra={"document_id": document_id, "error": str(exc)},
            )
            self._vector_repo.delete_by_document_id_metadata(document_id)
        logger.info("Vectors deleted", extra={"document_id": document_id})

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_document_id(document_id: str) -> None:
        """
        Reject falsy/placeholder document ids before they can reach chunking
        or storage.

        Catches: Python None passed where a str is expected (which str()
        elsewhere would silently turn into the literal text "None"), empty
        strings, and common placeholder values (including "string" — the
        Swagger/FastAPI default example value, which is easy to submit by
        accident from the auto-generated docs UI).
        """
        normalized = (document_id or "").strip().lower()
        if normalized in _INVALID_DOCUMENT_ID_VALUES:
            raise IngestionFailedError(
                str(document_id),
                f"Invalid or missing document_id: {document_id!r}. "
                "Refusing to ingest/delete — this would silently break every "
                "future document_id-scoped query for this document.",
            )
