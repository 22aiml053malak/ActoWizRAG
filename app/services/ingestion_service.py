"""
Ingestion service — orchestrates load → chunk → embed → store.

This service is called from the Celery task (ingestion_tasks.py).
It knows about repositories but never touches SQLAlchemy or PGVectorStore
directly — it delegates through the repository layer.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from llama_index.core.schema import BaseNode
from llama_index.readers.file import PDFReader

from app.core.config import settings
from app.core.exceptions import IngestionFailedError
from app.core.logger import get_logger
from app.repositories.vector_repository import VectorRepository
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.utils.file_storage import get_language

logger = get_logger(__name__)


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
    ) -> None:
        self._vector_repo = vector_repo
        self._chunking_service = chunking_service
        self._embedding_service = embedding_service

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
            IngestionFailedError on any unrecoverable failure.
        """
        logger.info(
            "Ingestion started",
            extra={
                "document_id": document_id,
                "filename": filename,
                "file_type": file_type,
            },
        )

        try:
            # ── 1. Load text ───────────────────────────────────────────────────
            text = self._load_text(storage_path, file_type)
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

            # Attach filename to every node's metadata for query-time attribution.
            for node in nodes:
                node.metadata["filename"] = filename

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

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_text(self, storage_path: str, file_type: str) -> str:
        path = Path(storage_path)
        if not path.exists():
            raise IngestionFailedError(str(path), f"File not found: {storage_path}")

        if file_type == "pdf":
            return self._load_pdf(storage_path)

        # All other types (text, markdown, code) are read as UTF-8 text.
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _load_pdf(storage_path: str) -> str:
        """Load PDF text using LlamaIndex's PDFReader."""
        try:
            reader = PDFReader()
            docs = reader.load_data(file=Path(storage_path))
            return "\n\n".join(d.get_content() for d in docs)
        except Exception:
            # Fallback to pypdf if PDFReader fails.
            import pypdf

            reader_pypdf = pypdf.PdfReader(storage_path)
            pages = [
                page.extract_text() or ""
                for page in reader_pypdf.pages
            ]
            return "\n\n".join(pages)

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
