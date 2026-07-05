"""
Re-ingest all documents with the new clean chunking pipeline.

This script:
  1. Reads all documents from the `documents` table that have status='completed'.
  2. Deletes their old vectors.
  3. Re-ingests them with the new ChunkingService (clean text, no HTML/OCR noise).

Run from the project root:
    source .venv/bin/activate
    python3 scripts/reingest_all.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import sqlalchemy as sa
from app.core.config import settings
from app.repositories.vector_repository import VectorRepository
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.document_loader_service import DocumentLoaderService
from app.services.ingestion_service import IngestionService


def main() -> None:
    sync_url = settings.DATABASE_SYNC_URL
    engine   = sa.create_engine(sync_url)

    with engine.connect() as conn:
        rows = conn.execute(sa.text(
            "SELECT id, filename, file_type, storage_path FROM documents WHERE status = 'completed'"
        )).fetchall()

    if not rows:
        print("No completed documents found — nothing to re-ingest.")
        return

    print(f"Found {len(rows)} document(s) to re-ingest.")

    vector_repo  = VectorRepository()
    chunking_svc = ChunkingService()
    embed_svc    = EmbeddingService()
    loader_svc   = DocumentLoaderService()
    ingest_svc   = IngestionService(
        vector_repo=vector_repo,
        chunking_service=chunking_svc,
        embedding_service=embed_svc,
        document_loader=loader_svc,
    )

    for doc_id, filename, file_type, storage_path in rows:
        doc_id_str = str(doc_id)
        print(f"\n[{filename}] id={doc_id_str}")

        print("  Deleting old vectors...")
        try:
            ingest_svc.delete(doc_id_str)
        except Exception as e:
            print(f"  WARNING: delete failed: {e}")

        print("  Re-ingesting...")
        try:
            count = ingest_svc.ingest(
                document_id=doc_id_str,
                storage_path=storage_path,
                filename=filename,
                file_type=file_type,
            )
            print(f"  OK — {count} chunks stored.")
        except Exception as e:
            print(f"  ERROR: {e}")

    engine.dispose()
    print("\nDone.")


if __name__ == "__main__":
    main()
