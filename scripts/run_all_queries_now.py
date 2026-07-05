#!/usr/bin/env python3
"""
Run all test questions end-to-end and write FINAL_RESULTS.txt.

Usage (from project root):
    python3 scripts/run_all_queries_now.py

Requires: Postgres running, pip install -r requirements.txt, alembic upgrade head
Does NOT require uvicorn or celery.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE = ROOT / "scripts" / "sample_assignment.txt"
OUT = ROOT / "FINAL_RESULTS.txt"

QUESTIONS = [
    {
        "query": "What is the goal of the assignment?",
        "keywords": ["evaluate", "api design", "rag", "semantic search"],
    },
    {
        "query": "What technologies are used for vector storage?",
        "keywords": ["postgresql", "pgvector"],
    },
    {
        "query": "How many developers will use the platform?",
        "keywords": ["100"],
    },
    {
        "query": "What is used for async document ingestion?",
        "keywords": ["celery", "redis"],
    },
]


def main() -> None:
    lines: list[str] = []
    lines.append(f"ActoWiz RAG — Final Results")
    lines.append(f"Generated: {datetime.now(tz=timezone.utc).isoformat()}")
    lines.append("=" * 72)

    from app.repositories.vector_repository import VectorRepository
    from app.services.chunking_service import ChunkingService
    from app.services.embedding_service import EmbeddingService
    from app.services.ingestion_service import IngestionService
    from app.services.llm_service import LLMService
    from app.services.rag_service import RAGService
    from app.services.retrieval_service import RetrievalService

    doc_id = str(uuid.uuid4())
    lines.append(f"Test document_id: {doc_id}")
    lines.append(f"Sample file: {SAMPLE}")
    lines.append("")

    vector_repo = VectorRepository()
    ingestion = IngestionService(
        vector_repo=vector_repo,
        chunking_service=ChunkingService(),
        embedding_service=EmbeddingService(),
    )
    rag = RAGService(
        retrieval_service=RetrievalService(
            vector_repo=vector_repo,
            embedding_service=EmbeddingService(),
        ),
        llm_service=LLMService(),
    )

    try:
        n = ingestion.ingest(
            document_id=doc_id,
            storage_path=str(SAMPLE),
            filename="sample_assignment.txt",
            file_type="text",
        )
        lines.append(f"Ingestion: OK ({n} chunks)")
    except Exception as exc:
        lines.append(f"Ingestion: FAILED — {exc}")
        OUT.write_text("\n".join(lines), encoding="utf-8")
        sys.exit(1)

    lines.append("")
    all_pass = True

    for i, q in enumerate(QUESTIONS, 1):
        lines.append(f"--- Question {i} ---")
        lines.append(f"Q: {q['query']}")
        try:
            resp = rag.query(
                query_text=q["query"],
                top_k=5,
                document_id=doc_id,
                generate_answer=True,
            )
            lines.append(f"Chunks retrieved: {len(resp.results)}")
            lines.append(f"Latency: {resp.latency_ms}ms")
            if resp.results:
                lines.append(f"Top chunk: {resp.results[0].content[:200]}...")
            lines.append(f"Answer: {resp.answer or '(none)'}")

            answer_lower = (resp.answer or "").lower()
            hits = [kw for kw in q["keywords"] if kw.lower() in answer_lower]
            passed = len(resp.results) > 0 and resp.answer and len(hits) >= 1
            lines.append(f"Keywords matched: {hits}")
            lines.append(f"RESULT: {'PASS' if passed else 'FAIL'}")
            if not passed:
                all_pass = False
        except Exception as exc:
            lines.append(f"RESULT: FAIL — {exc}")
            all_pass = False
        lines.append("")

    try:
        ingestion.delete(doc_id)
    except Exception:
        pass

    lines.append("=" * 72)
    lines.append(f"OVERALL: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
