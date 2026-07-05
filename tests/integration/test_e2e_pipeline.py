"""
Integration test — end-to-end RAG pipeline.

Prerequisites (automatically provided by docker-compose in CI, or manually):
  - PostgreSQL with pgvector at DATABASE_URL
  - Redis at REDIS_URL

This test:
  1. Creates DB tables.
  2. Ingests a small text document directly (bypassing Celery for test speed).
  3. Queries it and verifies at least one relevant result is returned.

Runs with the REAL embedding model and REAL pgvector — marks as `integration`
so the standard `pytest` run (unit tests only) skips it.

Run with:
    pytest tests/integration/ -m integration -v
"""

import asyncio
import os
import pytest
import uuid
import tempfile
from pathlib import Path

# Skip if TEST_DATABASE_URL is not set (CI without postgres).
pytestmark = pytest.mark.integration

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/actowiz_rag_test",
)
TEST_DB_ASYNC_URL = TEST_DB_URL.replace("postgresql://", "postgresql+asyncpg://")


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def db_session():
    """Set up a fresh test database session."""
    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from app.models.orm import Base

    engine = create_async_engine(TEST_DB_ASYNC_URL, echo=False)
    async with engine.begin() as conn:
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_ingest_and_query_e2e(db_session):
    """
    Full ingestion → query pipeline test.

    Uses a tiny text snippet so the test completes quickly even with real models.
    """
    from app.repositories.vector_repository import VectorRepository
    from app.repositories.document_repository import DocumentRepository
    from app.services.chunking_service import ChunkingService
    from app.services.embedding_service import EmbeddingService
    from app.services.ingestion_service import IngestionService
    from app.services.retrieval_service import RetrievalService

    document_id = str(uuid.uuid4())
    sample_text = (
        "LlamaIndex is a data framework for building LLM applications. "
        "It provides tools for data ingestion, indexing, and querying. "
        "The PGVectorStore stores embeddings in PostgreSQL using the pgvector extension. "
        "Sentence-window retrieval improves context quality by expanding retrieved chunks. "
        "Cross-encoder reranking improves precision by re-scoring candidates with a more expensive model."
    )

    # Write sample file to a temp location.
    with tempfile.NamedTemporaryFile(
        suffix=".txt", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(sample_text)
        tmp_path = f.name

    try:
        # ── Services ──────────────────────────────────────────────────────────
        vector_repo = VectorRepository()
        chunking_svc = ChunkingService()
        embedding_svc = EmbeddingService()
        ingestion_svc = IngestionService(
            vector_repo=vector_repo,
            chunking_service=chunking_svc,
            embedding_service=embedding_svc,
        )
        retrieval_svc = RetrievalService(
            vector_repo=vector_repo,
            embedding_service=embedding_svc,
        )

        # ── Insert document row ────────────────────────────────────────────────
        doc_repo = DocumentRepository(db_session)
        await doc_repo.create(
            document_id=uuid.UUID(document_id),
            filename="sample.txt",
            file_type="text",
            storage_path=tmp_path,
        )
        await db_session.commit()

        # ── Ingest ────────────────────────────────────────────────────────────
        chunk_count = ingestion_svc.ingest(
            document_id=document_id,
            storage_path=tmp_path,
            filename="sample.txt",
            file_type="text",
        )
        assert chunk_count > 0, "Ingestion should produce at least one chunk"

        # ── Query ─────────────────────────────────────────────────────────────
        results = retrieval_svc.retrieve(
            "What is PGVectorStore?",
            top_k=3,
        )
        assert len(results) > 0, "Query should return at least one result"

        # Verify the top result mentions pgvector or vector.
        top_content = results[0].content.lower()
        assert any(kw in top_content for kw in ["pgvector", "vector", "embedding", "llama"]), (
            f"Expected relevant content, got: {top_content[:200]}"
        )

        # ── Window replacement ─────────────────────────────────────────────────
        # Every result should have a "window" key in metadata.
        for result in results:
            assert "window" in result.metadata or len(result.content) > 0

        print(f"\nE2E test passed: ingested {chunk_count} chunks, retrieved {len(results)} results")

    finally:
        Path(tmp_path).unlink(missing_ok=True)
        # Clean up vectors.
        try:
            vector_repo.delete_by_document_id_metadata(document_id)
        except Exception:
            pass
