"""
Complete end-to-end test: upload → wait for ingestion → query
"""
import json
import sys
import time

import requests

API_BASE = "http://127.0.0.1:8000/api/v1"


def check_health():
    """Verify API is running."""
    try:
        resp = requests.get("http://127.0.0.1:8000/health", timeout=5)
        resp.raise_for_status()
        print("✓ API is healthy")
        return True
    except Exception as e:
        print(f"✗ API health check failed: {e}")
        print("  Make sure the API is running: uvicorn app.main:app --reload")
        return False


def upload_test_document():
    """Upload a test text file."""
    print("\n" + "=" * 80)
    print("STEP 1: Uploading test document")
    print("=" * 80)

    test_content = """
    ActoWiz Internal AI Knowledge Platform - Assignment Overview

    Goal of the Assignment:
    The goal of this assignment is to evaluate your understanding of API Design,
    RAG/Semantic Search Systems, AI Infrastructure, Database Design, Scalability
    & Reliability, and Production Engineering Practices.

    The platform allows internal developers to:
    1. Upload documents and code files
    2. Query documents using natural language
    3. Delete documents
    4. Perform semantic search over uploaded knowledge
    5. Access LLMs through a centralized AI Gateway

    The platform will be used internally by approximately 100 developers.

    Key Features:
    - Document upload API with async processing
    - Semantic search with vector embeddings
    - LLM-powered answer generation via Groq
    - PostgreSQL with pgvector for vector storage
    - Celery for async task processing
    """

    with open("temp_test_doc.txt", "w", encoding="utf-8") as f:
        f.write(test_content)

    try:
        with open("temp_test_doc.txt", "rb") as f:
            resp = requests.post(
                f"{API_BASE}/documents",
                files={"file": ("assignment_info.txt", f, "text/plain")},
                timeout=30,
            )
        resp.raise_for_status()
        data = resp.json()
        print(f"✓ Document uploaded")
        print(f"  Document ID: {data['document_id']}")
        print(f"  Status: {data['status']}")
        return data["document_id"]
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        return None


def wait_for_ingestion(document_id, max_wait=120):
    """Poll document status until completed or timed out."""
    print("\n" + "=" * 80)
    print("STEP 2: Waiting for ingestion to complete")
    print("=" * 80)

    start_time = time.time()
    last_status = None

    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(f"{API_BASE}/documents/{document_id}", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            status = data["status"]

            if status != last_status:
                print(
                    f"  Status: {status} (chunk_count: {data.get('chunk_count', 0)})"
                )
                last_status = status

            if status == "completed":
                print(
                    f"✓ Ingestion completed in {int(time.time() - start_time)}s"
                )
                print(f"  Chunks created: {data['chunk_count']}")
                return True

            if status == "failed":
                print(f"✗ Ingestion failed: {data.get('error_message')}")
                return False

            time.sleep(2)
        except Exception as e:
            print(f"  Error checking status: {e}")
            time.sleep(2)

    print(f"✗ Timeout waiting for ingestion ({max_wait}s)")
    print("  Possible issues:")
    print("    - Celery worker not running")
    print("    - Redis not running")
    print("  Check worker logs: docker compose logs worker")
    return False


def test_query(query_text):
    """Test semantic search without LLM answer generation."""
    print("\n" + "=" * 80)
    print("STEP 3: Testing semantic search")
    print("=" * 80)
    print(f"Query: '{query_text}'")

    payload = {
        "query": query_text,
        "top_k": 5,
        "generate_answer": False,
    }

    try:
        resp = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        print(f"\nResults found: {len(data['results'])}")
        print(f"Latency: {data['latency_ms']}ms")

        if data["results"]:
            print("\n✓ Query successful!")
            for i, result in enumerate(data["results"][:2], 1):
                print(f"\n  Result {i}:")
                print(f"    Score: {result['score']:.4f}")
                print(f"    Filename: {result['filename']}")
                print(f"    Content preview: {result['content'][:150]}...")
            return True

        print("\n✗ Query returned no results")
        print("  This means:")
        print("    - Documents exist in DB but semantic match found nothing")
        print("    - Or embeddings weren't created properly")
        return False
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False


def test_query_with_answer(query_text):
    """Test query with LLM answer generation."""
    print("\n" + "=" * 80)
    print("STEP 4: Testing with LLM answer generation")
    print("=" * 80)
    print(f"Query: '{query_text}'")

    payload = {
        "query": query_text,
        "top_k": 5,
        "generate_answer": True,
    }

    try:
        resp = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        print(f"\nResults found: {len(data['results'])}")
        print(f"Latency: {data['latency_ms']}ms")

        if data.get("answer"):
            print(f"\n✓ LLM Answer:\n{data['answer']}")
            if data.get("sources"):
                print(f"\nSources: {', '.join(data['sources'])}")
            return True

        print("\n✗ No answer generated")
        return False
    except Exception as e:
        print(f"✗ Query with answer failed: {e}")
        return False


def main():
    print("ActoWiz RAG - Complete Flow Test")
    print("=" * 80)

    if not check_health():
        sys.exit(1)

    document_id = upload_test_document()
    if not document_id:
        sys.exit(1)

    if not wait_for_ingestion(document_id):
        sys.exit(1)

    query = "what is the goal of the assignment?"
    if not test_query(query):
        sys.exit(1)

    if not test_query_with_answer(query):
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
