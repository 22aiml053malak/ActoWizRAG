#!/usr/bin/env python3
"""
Validate the full RAG pipeline for LOCAL dev (uvicorn + celery, no Docker API).

Run from project root BEFORE starting servers:
    python3 scripts/validate_pipeline.py

Run AFTER uploading a document via Swagger (checks API + Celery path):
    python3 scripts/validate_pipeline.py --via-api

This script checks .env, DB, Redis, ingestion, retrieval, and LLM answers.
"""
from __future__ import annotations

import argparse
import socket
import sys
import tempfile
import uuid
from pathlib import Path

# Ensure project root is on sys.path when run as scripts/validate_pipeline.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_PATH = ROOT / "scripts" / "sample_assignment.txt"

# Questions derived from scripts/sample_assignment.txt — keyword checks are case-insensitive.
TEST_QUESTIONS = [
    {
        "query": "What is the goal of the assignment?",
        "keywords": ["evaluate", "api design", "rag", "semantic search", "production engineering"],
        "min_keyword_hits": 2,
    },
    {
        "query": "What technologies are used for vector storage?",
        "keywords": ["postgresql", "pgvector", "vector"],
        "min_keyword_hits": 1,
    },
    {
        "query": "How many developers will use the platform?",
        "keywords": ["100", "developers"],
        "min_keyword_hits": 1,
    },
    {
        "query": "What is used for async document ingestion?",
        "keywords": ["celery", "redis", "async"],
        "min_keyword_hits": 1,
    },
]


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def check_env() -> bool:
    print("\n[1/6] Checking .env / settings")
    from app.core.config import settings

    ok = True
    _ok(f"DATABASE_SYNC_URL host reachable config loaded")
    print(f"      DATABASE_SYNC_URL = {settings.DATABASE_SYNC_URL}")
    print(f"      REDIS_URL         = {settings.REDIS_URL}")
    print(f"      EMBED_MODEL       = {settings.EMBED_MODEL_NAME} (dim={settings.EMBED_DIM})")
    print(f"      LLM_PROVIDER      = {settings.LLM_PROVIDER} / {settings.LLM_MODEL_NAME}")

    if "127.0.0.1" in settings.DATABASE_SYNC_URL or "localhost" in settings.DATABASE_SYNC_URL:
        _ok("Database URL uses localhost — correct for local uvicorn + celery")
    elif "@postgres:" in settings.DATABASE_SYNC_URL:
        _fail("DATABASE_URL uses Docker hostname 'postgres' — change to 127.0.0.1 for local dev")
        ok = False

    if "redis://redis:" in settings.REDIS_URL:
        _fail("REDIS_URL uses Docker hostname 'redis' — change to redis://localhost:6379/0")
        ok = False
    else:
        _ok("Redis URL uses localhost — correct for local celery")

    if not settings.LLM_API_KEY:
        _fail("LLM_API_KEY is empty — answers will not be generated")
        ok = False
    else:
        _ok("LLM_API_KEY is set")

    if settings.LLM_PROVIDER == "none":
        _fail("LLM_PROVIDER=none — set LLM_PROVIDER=groq in .env")
        ok = False

    return ok


def check_tcp(host: str, port: int, label: str) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            _ok(f"{label} reachable at {host}:{port}")
            return True
    except OSError as exc:
        _fail(f"{label} NOT reachable at {host}:{port} ({exc})")
        return False


def check_connectivity() -> bool:
    print("\n[2/6] Checking Postgres + Redis connectivity")
    pg_ok = check_tcp("127.0.0.1", 5432, "Postgres")
    redis_ok = check_tcp("127.0.0.1", 6379, "Redis")
    if not pg_ok:
        print("      Tip: docker compose up postgres -d   OR start local Postgres")
    if not redis_ok:
        print("      Tip: docker compose up redis -d      OR start local Redis")
    return pg_ok and redis_ok


def check_database() -> bool:
    print("\n[3/6] Checking database schema + vector chunks")
    import sqlalchemy as sa

    from app.core.config import settings
    from app.repositories.vector_repository import PHYSICAL_CHUNK_TABLE_NAME

    try:
        engine = sa.create_engine(settings.DATABASE_SYNC_URL)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
            _ok("Database connection successful")

            ext = conn.execute(
                sa.text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            ).fetchone()
            if ext:
                _ok("pgvector extension installed")
            else:
                _fail("pgvector extension missing — run: CREATE EXTENSION vector;")
                return False

            for table in ("documents", "query_logs"):
                exists = conn.execute(
                    sa.text(
                        "SELECT EXISTS (SELECT FROM information_schema.tables "
                        "WHERE table_schema='public' AND table_name=:t)"
                    ),
                    {"t": table},
                ).scalar()
                if exists:
                    _ok(f"Table '{table}' exists")
                else:
                    _fail(f"Table '{table}' missing — run: alembic upgrade head")
                    return False

            chunk_exists = conn.execute(
                sa.text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables "
                    "WHERE table_schema='public' AND table_name=:t)"
                ),
                {"t": PHYSICAL_CHUNK_TABLE_NAME},
            ).scalar()
            if chunk_exists:
                count = conn.execute(
                    sa.text(f"SELECT COUNT(*) FROM {PHYSICAL_CHUNK_TABLE_NAME}")
                ).scalar()
                _ok(f"Vector table '{PHYSICAL_CHUNK_TABLE_NAME}' has {count} chunks")
            else:
                print(f"      (Vector table '{PHYSICAL_CHUNK_TABLE_NAME}' not yet created — "
                      "will be created on first ingestion)")

        engine.dispose()
        return True
    except Exception as exc:
        _fail(f"Database error: {exc}")
        print("      Tip: verify postgres/postgres credentials in .env match your Postgres")
        return False


def _score_answer(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in lower)


def run_direct_pipeline() -> bool:
    """Ingest sample doc directly and run test queries (no API/celery needed)."""
    print("\n[4/6] Direct pipeline test (ingest → retrieve → LLM answer)")

    if not SAMPLE_PATH.exists():
        _fail(f"Sample file missing: {SAMPLE_PATH}")
        return False

    from app.repositories.vector_repository import VectorRepository
    from app.services.chunking_service import ChunkingService
    from app.services.embedding_service import EmbeddingService
    from app.services.ingestion_service import IngestionService
    from app.services.llm_service import LLMService
    from app.services.rag_service import RAGService
    from app.services.retrieval_service import RetrievalService

    document_id = str(uuid.uuid4())
    print(f"      Test document_id: {document_id}")

    vector_repo = VectorRepository()
    ingestion_svc = IngestionService(
        vector_repo=vector_repo,
        chunking_service=ChunkingService(),
        embedding_service=EmbeddingService(),
    )
    retrieval_svc = RetrievalService(
        vector_repo=vector_repo,
        embedding_service=EmbeddingService(),
    )
    rag_svc = RAGService(retrieval_service=retrieval_svc, llm_service=LLMService())

    try:
        chunk_count = ingestion_svc.ingest(
            document_id=document_id,
            storage_path=str(SAMPLE_PATH),
            filename="sample_assignment.txt",
            file_type="text",
        )
        _ok(f"Ingested {chunk_count} chunks")
    except Exception as exc:
        _fail(f"Ingestion failed: {exc}")
        return False

    all_pass = True
    print("\n[5/6] Test questions (direct pipeline)")
    for i, test in enumerate(TEST_QUESTIONS, 1):
        q = test["query"]
        print(f"\n  Q{i}: {q}")
        try:
            response = rag_svc.query(
                query_text=q,
                top_k=5,
                document_id=document_id,
                generate_answer=True,
            )
        except Exception as exc:
            _fail(f"Query failed: {exc}")
            all_pass = False
            continue

        print(f"      Results: {len(response.results)} chunks | Latency: {response.latency_ms}ms")

        if not response.results:
            _fail("No chunks retrieved — embedding/search issue")
            all_pass = False
            continue

        top = response.results[0].content[:120].replace("\n", " ")
        print(f"      Top chunk: {top}...")

        if not response.answer:
            _fail("No LLM answer returned")
            all_pass = False
            continue

        hits = _score_answer(response.answer, test["keywords"])
        min_hits = test["min_keyword_hits"]
        print(f"      Answer: {response.answer[:200]}{'...' if len(response.answer) > 200 else ''}")

        if hits >= min_hits:
            _ok(f"Answer quality OK ({hits}/{len(test['keywords'])} keywords matched, need {min_hits})")
        else:
            _fail(f"Answer quality LOW ({hits}/{len(test['keywords'])} keywords matched, need {min_hits})")
            all_pass = False

    # Cleanup test vectors
    try:
        ingestion_svc.delete(document_id)
    except Exception:
        pass

    return all_pass


def run_via_api() -> bool:
    """Test via running API (Swagger path: upload handled separately)."""
    print("\n[4/6] API pipeline test (requires uvicorn on :8000)")

    try:
        import requests
    except ImportError:
        _fail("requests not installed — pip install requests")
        return False

    try:
        health = requests.get("http://127.0.0.1:8000/health", timeout=5).json()
        _ok(f"API health: {health.get('status')} | chunks={health.get('stored_chunks')}")
    except Exception as exc:
        _fail(f"API not reachable: {exc}")
        print("      Start: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return False

    # Upload sample doc
    with open(SAMPLE_PATH, "rb") as f:
        resp = requests.post(
            "http://127.0.0.1:8000/api/v1/documents",
            files={"file": ("sample_assignment.txt", f, "text/plain")},
            timeout=30,
        )
    if resp.status_code != 202:
        _fail(f"Upload failed: {resp.status_code} {resp.text}")
        return False

    doc_id = resp.json()["document_id"]
    _ok(f"Uploaded document {doc_id}")

    # Wait for celery ingestion
    import time

    print("      Waiting for Celery ingestion (max 120s)...")
    for _ in range(60):
        status_resp = requests.get(
            f"http://127.0.0.1:8000/api/v1/documents/{doc_id}", timeout=5
        ).json()
        status = status_resp["status"]
        if status == "completed":
            _ok(f"Ingestion completed ({status_resp['chunk_count']} chunks)")
            break
        if status == "failed":
            _fail(f"Ingestion failed: {status_resp.get('error_message')}")
            return False
        time.sleep(2)
    else:
        _fail("Ingestion timed out — is Celery worker running?")
        print("      Start: celery -A app.workers.celery_app:celery_app worker --loglevel=info")
        return False

    all_pass = True
    print("\n[5/6] Test questions (via API / Swagger path)")
    for i, test in enumerate(TEST_QUESTIONS, 1):
        q = test["query"]
        print(f"\n  Q{i}: {q}")
        resp = requests.post(
            "http://127.0.0.1:8000/api/v1/query",
            json={"query": q, "top_k": 5, "generate_answer": True, "document_id": doc_id},
            timeout=120,
        )
        if resp.status_code != 200:
            _fail(f"Query HTTP {resp.status_code}: {resp.text[:200]}")
            all_pass = False
            continue

        data = resp.json()
        print(f"      Results: {len(data['results'])} | Latency: {data['latency_ms']}ms")

        if not data["results"]:
            _fail("No chunks retrieved")
            all_pass = False
            continue

        answer = data.get("answer") or ""
        if not answer:
            _fail("No LLM answer")
            all_pass = False
            continue

        hits = _score_answer(answer, test["keywords"])
        print(f"      Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        if hits >= test["min_keyword_hits"]:
            _ok(f"Answer quality OK ({hits} keyword hits)")
        else:
            _fail(f"Answer quality LOW ({hits} keyword hits, need {test['min_keyword_hits']})")
            all_pass = False

    return all_pass


def print_swagger_guide() -> None:
    print("\n[6/6] Swagger UI testing guide (local uvicorn + celery)")
    print("""
  Prerequisites (one-time):
    alembic upgrade head
    # Postgres + Redis must be running on localhost (docker compose up postgres redis -d is fine)

  Terminal 1 — Celery worker:
    cd /home/malak/projects/ActoWizRAG
    celery -A app.workers.celery_app:celery_app worker --loglevel=info

  Terminal 2 — API server:
    cd /home/malak/projects/ActoWizRAG
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

  Swagger: http://127.0.0.1:8000/docs

  Step 1 — Upload document:
    POST /api/v1/documents  →  upload scripts/sample_assignment.txt

  Step 2 — Wait for ingestion:
    GET /api/v1/documents/{document_id}  →  status must be "completed"

  Step 3 — Query (generate_answer defaults to true):
    POST /api/v1/query
    {
      "query": "What is the goal of the assignment?",
      "top_k": 5,
      "generate_answer": true
    }

  Expected: non-empty "results" array AND a meaningful "answer" string.
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ActoWiz RAG local pipeline")
    parser.add_argument(
        "--via-api",
        action="store_true",
        help="Test via running API + Celery (Swagger path) instead of direct ingestion",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("ActoWiz RAG — Local Pipeline Validation")
    print("=" * 72)

    steps_ok = check_env()
    steps_ok = check_connectivity() and steps_ok
    steps_ok = check_database() and steps_ok

    if not steps_ok:
        print("\n✗ FIX the issues above before testing in Swagger.")
        print_swagger_guide()
        sys.exit(1)

    pipeline_ok = run_via_api() if args.via_api else run_direct_pipeline()
    print_swagger_guide()

    if pipeline_ok:
        print("\n" + "=" * 72)
        print("✓ ALL PIPELINE TESTS PASSED — ready for Swagger UI testing")
        print("=" * 72)
        sys.exit(0)

    print("\n" + "=" * 72)
    print("✗ PIPELINE TESTS FAILED — fix issues above before Swagger testing")
    print("=" * 72)
    sys.exit(1)


if __name__ == "__main__":
    main()
