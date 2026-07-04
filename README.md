# ActoWiz Internal AI Knowledge Platform — RAG Backend

Production-grade backend for an internal developer knowledge search system. Built with FastAPI, Celery, PostgreSQL + pgvector, LlamaIndex, and Groq LLM.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack](#tech-stack)
3. [Database Schema](#database-schema)
4. [Chunking Strategy](#chunking-strategy)
5. [Retrieval Pipeline](#retrieval-pipeline)
6. [AI Gateway (Groq)](#ai-gateway-groq)
7. [Quick Start](#quick-start)
8. [Running Without Docker (Local Dev)](#running-without-docker-local-dev)
9. [Alembic Migration Commands](#alembic-migration-commands)
10. [API Reference](#api-reference)
11. [Testing](#testing)
12. [Scaling Strategy](#scaling-strategy)
13. [Trade-offs & Assumptions](#trade-offs--assumptions)
14. [Acceptance Checklist](#acceptance-checklist)

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Ingestion ["📥 Ingestion Pipeline"]
        UP[POST /documents\nmultipart upload] --> FS[Save file to disk\nfile_storage.py]
        FS --> DB1[Insert documents row\nstatus=pending]
        DB1 --> CQ[Enqueue Celery task\ningest_document_task]
        CQ --> W1[Celery Worker]
        W1 --> LOAD[Load text\nPDFReader / plain text]
        LOAD --> CHUNK{File type?}
        CHUNK -->|prose / PDF / MD| SWP[SentenceWindowNodeParser\nwindow_size=3]
        CHUNK -->|code .py .js …| CS[CodeSplitter\n40 lines, 15 overlap]
        CS --> WA[Window augmentation\nprev + curr + next chunk]
        SWP --> EMB[HuggingFaceEmbedding\nall-MiniLM-L6-v2\nembed_dim=384]
        WA --> EMB
        EMB --> PGV[PGVectorStore.add\ndocument_chunks table\nvector 384]
        PGV --> DB2[Update status=completed\nchunk_count=N]
    end

    subgraph Query ["🔍 Query Pipeline"]
        QR[POST /query] --> QE[Embed query\nsame HuggingFaceEmbedding]
        QE --> VS[Vector search\nsimilarity_top_k=15\nMetadataFilters]
        VS --> RR[Cross-encoder rerank\nms-marco-MiniLM-L-6-v2\ntop_n=top_k]
        RR --> WR[MetadataReplacementPostProcessor\nswap sentence → window text]
        WR --> RESP[Return ChunkResult list\n+ scores + metadata]
        RESP -->|generate_answer=true| LLM[Groq AI Gateway\nllama3-8b-8192]
        LLM --> ANS[Return answer + sources]
        RESP --> LOG[Async: log to query_logs]
    end

    subgraph Delete ["🗑️ Delete Pipeline"]
        DEL[DELETE /documents/id] --> SD[Soft-delete documents row\nstatus=deleting\ndeleted_at=now]
        SD --> DQ[Enqueue delete_document_task]
        DQ --> W2[Celery Worker]
        W2 --> VD[Hard-delete vectors\nPGVectorStore metadata filter]
        VD --> HD[Hard-delete documents row]
    end

    subgraph Migrations ["🗄️ Schema Management"]
        MIG[alembic upgrade head] --> PGX[CREATE EXTENSION IF NOT EXISTS vector]
        PGX --> T1[CREATE TABLE documents]
        PGX --> T2[CREATE TABLE query_logs]
        PGV2[PGVectorStore first use] --> T3[CREATE TABLE document_chunks\nvector 384 column\nHNSW index]
    end
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| API | FastAPI + Pydantic v2 |
| Async tasks | Celery 5 + Redis |
| Database | PostgreSQL 16 + pgvector extension (your own instance) |
| ORM | SQLAlchemy 2.x (async) |
| Migrations | **Alembic** |
| Vector store | LlamaIndex `PGVectorStore` (same Postgres DB) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Chunking — prose | `SentenceWindowNodeParser` |
| Chunking — code | `CodeSplitter` (tree-sitter) + window augmentation |
| PDF parsing | LlamaIndex `PDFReader` (fallback: `pypdf`) |
| **LLM / AI Gateway** | **Groq Cloud** (`llama3-8b-8192`) via OpenAI-compatible API |
| Container | Docker + docker-compose |

---

## Database Schema

### Custom tables (managed by Alembic)

#### `documents`

```sql
CREATE TABLE documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    VARCHAR(512)  NOT NULL,
    file_type   VARCHAR(50)   NOT NULL,   -- 'pdf' | 'code' | 'text' | 'markdown'
    storage_path VARCHAR(1024) NOT NULL,
    status      VARCHAR(20)   NOT NULL DEFAULT 'pending',
                                          -- pending|processing|completed|failed|deleting
    error_message TEXT,
    chunk_count INTEGER DEFAULT 0,
    uploaded_at TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ   NOT NULL DEFAULT now(),
    deleted_at  TIMESTAMPTZ             -- soft delete marker
);

-- Partial index — only live documents (not soft-deleted)
CREATE INDEX idx_documents_status
    ON documents(status)
    WHERE deleted_at IS NULL;
```

#### `query_logs`

```sql
CREATE TABLE query_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text      TEXT    NOT NULL,
    filters         JSONB,
    top_k           INTEGER,
    result_chunk_ids JSONB,   -- [{node_id, score}, ...]
    latency_ms      INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_query_logs_created_at
    ON query_logs(created_at DESC);
```

### Chunk embeddings table (auto-created by LlamaIndex `PGVectorStore`)

LlamaIndex materialises the chunk+embedding table **automatically on first ingestion** — it is a real Postgres table you can inspect:

```sql
-- After first document is ingested, inspect the table:
\d document_chunks
```

Approximate schema (generated by `PGVectorStore.from_params(table_name="document_chunks", embed_dim=384)`):

```
                    Table "public.document_chunks"
    Column    │         Type          │ Nullable
──────────────┼───────────────────────┼──────────
 id           │ text                  │ not null   ← LlamaIndex node UUID
 text         │ text                  │            ← node text content
 metadata_    │ jsonb                 │            ← document_id, filename, window, …
 node_info    │ jsonb                 │            ← start/end char positions
 embedding    │ vector(384)           │            ← 384-dim float vector

Indexes:
    "document_chunks_pkey" PRIMARY KEY (id)
    "document_chunks_embedding_idx" hnsw (embedding vector_cosine_ops)
```

**Important**: `embed_dim=384` is set explicitly in `vector_repository.py` and matches the `all-MiniLM-L6-v2` model output. A mismatch causes `"different vector dimensions"` errors.

Every chunk's `metadata_` JSONB contains:

| Key | Value |
|-----|-------|
| `document_id` | UUID matching `documents.id` |
| `file_type` | `pdf` / `code` / `text` / `markdown` |
| `filename` | Original filename |
| `window` | Surrounding sentence/code context |
| `original_text` | Tight embedded text |

---

## Chunking Strategy

| File type | Strategy | Rationale |
|-----------|----------|-----------|
| Text / PDF / Markdown | `SentenceWindowNodeParser` | Prose has semantic structure at the sentence level. Embedding individual sentences gives **precision**. Surrounding ±3 sentences stored in `window` metadata gives **context** at retrieval time without degrading embedding quality. |
| Code | `CodeSplitter` + window augmentation | Tree-sitter parses AST boundaries (function/class edges). Each chunk's `window` = prev+curr+next chunk text, mirroring sentence-window: **embed tight, serve wide**. |

`MetadataReplacementPostProcessor` swaps the embedded sentence for the window text before returning results — best of both worlds at zero extra embedding cost.

---

## Retrieval Pipeline

```
Query string
    │
    ▼
HuggingFaceEmbedding.get_query_embedding()      ← same model as ingestion (384-dim)
    │
    ▼
PGVectorStore.query(similarity_top_k=15)         ← wide cosine candidate set
    │   MetadataFilters if document_id/filters provided
    │
    ▼
SentenceTransformerRerank(top_n=top_k)           ← cross-encoder re-scores 15 → top_k
    │   ms-marco-MiniLM-L-6-v2
    │
    ▼
MetadataReplacementPostProcessor(key="window")   ← swap sentence → window text
    │
    ▼
List[ChunkResult]  → return / pass to Groq
```

---

## AI Gateway (Groq)

`llm_service.py` implements a provider-agnostic protocol. The active provider is selected via `LLM_PROVIDER` env var.

**Default provider: Groq** — fast inference on Meta Llama 3 models.

Supported providers:

| `LLM_PROVIDER` | Description |
|---------------|-------------|
| `groq` | Groq Cloud (default). Fast, free tier available. |
| `openai_compatible` | Any OpenAI-compatible endpoint (OpenAI, Azure, Together, etc.) |
| `none` | Disabled. Returns chunks only, no answer synthesis. |

**Recommended Groq models:**

| Model | Context | Speed | Use case |
|-------|---------|-------|---------|
| `llama3-8b-8192` | 8K | Fastest | Default; great for Q&A over chunks |
| `llama3-70b-8192` | 8K | Fast | Higher quality answers |
| `mixtral-8x7b-32768` | 32K | Medium | Long document context |
| `gemma2-9b-it` | 8K | Fast | Instruction-following |

Get a free Groq API key at: https://console.groq.com

---

## Quick Start

### Prerequisites

- Docker ≥ 24 and Docker Compose v2
- A PostgreSQL instance with the **pgvector extension** installed
  - Docker option: `pgvector/pgvector:pg16` image (handled automatically if you use the bundled postgres service)
  - External Postgres: run `CREATE EXTENSION IF NOT EXISTS vector;` in your database
- A Groq API key (free at https://console.groq.com) for answer generation

### 1. Clone & configure

```bash
git clone <repo-url>
cd ActoWizRAG

# Copy the example env file
cp .env.example .env

# Edit .env — fill in your actual values:
#   DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
#   DATABASE_SYNC_URL=postgresql://user:pass@host:5432/dbname
#   LLM_API_KEY=gsk_your_groq_api_key_here
nano .env  # or vim / code .env
```

### 2. Start all services with Docker

```bash
docker compose up --build -d
```

This starts (in order):
1. **postgres** — PostgreSQL 16 with pgvector _(skip if using external Postgres)_
2. **redis** — Message broker + Celery result backend
3. **migrate** — Runs `alembic upgrade head` to create all tables, then exits
4. **api** — FastAPI on http://localhost:8000
5. **worker** — Celery worker for background ingestion

Check all services are healthy:

```bash
docker compose ps
curl http://localhost:8000/health
# → {"status": "ok", "version": "1.0.0"}
```

### 3. Upload a PDF

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@/path/to/your_document.pdf"
# → {"document_id": "550e8400-...", "status": "pending", "message": "Document accepted for ingestion"}
```

### 4. Poll ingestion status

```bash
curl http://localhost:8000/api/v1/documents/550e8400-...
# → {"status": "processing", ...}
# → {"status": "completed", "chunk_count": 87, ...}
```

### 5. Upload a Python file

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@/path/to/script.py"
```

### 6. Semantic query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does the authentication system work?",
    "top_k": 5
  }'
```

### 7. Query with metadata filter (specific document)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "connection pooling",
    "top_k": 3,
    "document_id": "550e8400-..."
  }'
```

### 8. Generate an answer via Groq

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main components of the system?",
    "top_k": 5,
    "generate_answer": true
  }'
# → {"results": [...], "answer": "The system consists of...", "sources": ["report.pdf"]}
```

### 9. Delete a document

```bash
curl -X DELETE http://localhost:8000/api/v1/documents/550e8400-...
# → {"status": "deleting", "message": "Document marked for deletion. Vectors will be removed asynchronously."}
```

### Interactive API docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Running Without Docker (Local Dev)

### Prerequisites

- Python 3.11+
- PostgreSQL with pgvector (`CREATE EXTENSION IF NOT EXISTS vector;`)
- Redis running locally (`redis-server` or via Docker)

### Steps

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your DB URLs and Groq API key

# 4. Run Alembic migrations (creates tables)
alembic upgrade head

# 5. Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 6. In a separate terminal, start the Celery worker
celery -A app.workers.celery_app:celery_app worker \
  --loglevel=info \
  --concurrency=2 \
  --prefetch-multiplier=1
```

---

## Alembic Migration Commands

All Alembic commands should be run from the project root where `alembic.ini` lives.
The `DATABASE_SYNC_URL` env var (from `.env`) is automatically used.

```bash
# Apply all pending migrations (run this before first start)
alembic upgrade head

# Check current migration state
alembic current

# Show full migration history
alembic history --verbose

# Roll back the most recent migration
alembic downgrade -1

# Roll back ALL migrations (drops all tables)
alembic downgrade base

# Auto-generate a new migration from ORM model changes
# (after editing app/models/orm.py)
alembic revision --autogenerate -m "add new column to documents"

# Generate SQL for a migration without applying it (dry run)
alembic upgrade head --sql

# Stamp the database as being at a specific revision
# (use when the DB was created manually and you want Alembic to track it)
alembic stamp head
```

> **Note**: The `document_chunks` table is **not** managed by Alembic. It is created automatically by LlamaIndex's `PGVectorStore` the first time a document is ingested. You can inspect it with `\d document_chunks` in psql.

---

## API Reference

### `POST /api/v1/documents` — Upload document

Upload a file for async ingestion. Returns 202 immediately.

**Accepted types:** `.pdf`, `.txt`, `.md`, `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.cpp`, `.c`, `.cs`, `.rb`, `.php`, `.sh`, `.yaml`, `.yml`, `.json`, `.toml`

```json
// Response 202
{"document_id": "uuid", "status": "pending", "message": "Document accepted for ingestion"}
```

### `GET /api/v1/documents` — List documents

Query params: `limit` (default 50), `offset` (default 0)

### `GET /api/v1/documents/{id}` — Get status

Status values: `pending` → `processing` → `completed` | `failed` | `deleting`

### `DELETE /api/v1/documents/{id}` — Delete document

Soft-deletes immediately, hard-deletes vectors via Celery.

### `POST /api/v1/query` — Semantic search

```json
// Request
{
  "query": "string (required, 1-4096 chars)",
  "top_k": 5,
  "filters": {"file_type": "pdf"},
  "document_id": "optional-uuid",
  "generate_answer": false
}

// Response
{
  "query": "...",
  "top_k": 5,
  "results": [
    {
      "node_id": "...",
      "document_id": "...",
      "filename": "report.pdf",
      "content": "windowed context text",
      "score": 0.92,
      "metadata": {}
    }
  ],
  "answer": "Groq LLM answer (only when generate_answer=true)",
  "sources": ["report.pdf"],
  "latency_ms": 312
}
```

### `GET /health` — Health check

```json
{"status": "ok", "version": "1.0.0"}
```

---

## Testing

### Unit tests (fast, no external services required)

```bash
# Install dependencies
pip install -r requirements.txt

# Run unit tests (uses mocked embedding/reranker — no GPU/DB needed)
pytest tests/unit/ -v
```

Unit test coverage:
- `test_chunking_service.py` — prose window metadata, code window augmentation, fallback on unknown language
- `test_embedding_service.py` — 384-dim output, batch embedding count
- `test_retrieval_service.py` — rerank changes ordering, empty candidates, document_id filter forwarding
- `test_retrieval_service.py::TestDeletePartialFailure` — fallback metadata delete on primary failure

### Integration tests (requires live Postgres + Redis)

```bash
# With Postgres running (docker-compose or local):
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/actowiz_rag \
pytest tests/integration/ -m integration -v
```

---

## Scaling Strategy

### Ingestion throughput
- **Horizontal Celery workers**: Scale `worker` replicas. Each process loads the embedding model independently. CPU-bound; size by CPU cores.
- **Batched embedding**: `embed_batch_size=32` amortizes model overhead. At higher volume, move to a dedicated GPU embedding microservice.
- **Non-blocking upload**: API returns 202 immediately; ingestion never blocks the request thread.

### Query latency
- **HNSW index on pgvector**: Sub-linear ANN search. Tune `m` (connectivity) and `ef_construction` (recall) for your recall/speed target.
- **IVFFlat alternative**: Better for very large tables (>1M vectors) where HNSW memory is prohibitive.
- **Redis query caching**: Cache `hash(query + filters + top_k)` with 5–60 min TTL.
- **Read replicas**: Route `/query` traffic to Postgres read replicas; ingestion writes go to primary.
- **Dedicated vector DB graduation**: At ~10M+ vectors, migrate `VectorRepository` to Qdrant or Milvus. The abstraction layer makes this a single-file change.

### Model serving
- At scale, run `all-MiniLM-L6-v2` and the cross-encoder behind Triton Inference Server or FastEmbed for batching, GPU acceleration, and model versioning.

---

## Trade-offs & Assumptions

### pgvector-in-Postgres vs. dedicated vector DB

**Choice**: pgvector in the same Postgres instance.

**Rationale**: Single datastore = one backup, one connection pool, ACID transactions across metadata + vectors. At ~100 internal developers, pgvector with HNSW keeps p95 search < 50ms. Graduate to Qdrant/Milvus when the `document_chunks` table exceeds ~5–10M vectors.

### Soft delete vs. hard delete

| Table | Delete type | Reason |
|-------|-------------|--------|
| `documents` | **Soft** (`deleted_at`, `status='deleting'`) | Audit trail, prevent race conditions with in-flight Celery tasks, enable restore |
| `document_chunks` | **Hard** (via Celery task) | No compliance reason to retain vectors; re-running the delete is idempotent |

### Alembic for migrations

All schema changes go through Alembic revision files (`alembic/versions/`). The application verifies DB connectivity on startup but does **not** auto-create or modify tables at runtime. This follows the "migrations as code" principle: schema changes are tracked in git, reviewed, and applied explicitly.

### Groq as the LLM provider

**Choice**: Groq Cloud (`llama3-8b-8192`) as the default LLM.

**Rationale**: Groq provides OpenAI-compatible APIs with industry-leading inference speed (hundreds of tokens/second), a generous free tier, and no self-hosting complexity. Switching to another provider (OpenAI, Azure, Together) only requires changing `LLM_PROVIDER` and `LLM_API_KEY` in `.env` — no code changes needed.

### Synchronous embedding in Celery worker

Embedding is CPU-bound and runs synchronously inside the Celery task. At 100-developer scale this is fine (`prefetch_multiplier=1` ensures one task per worker). At higher volume (>1000 docs/hour), move to async micro-batched embedding with a dedicated service.

### Local sentence-transformers over hosted APIs

- **Cost**: Zero per-request embedding cost
- **Latency**: No network round-trip
- **Privacy**: Documents never leave your infrastructure
- **Trade-off**: Requires adequate CPU/RAM (`all-MiniLM-L6-v2` ≈ 80MB, fast inference)

### tree-sitter-language-pack

The `tree_sitter_languages` package was renamed to `tree-sitter-language-pack`. `requirements.txt` pins `tree-sitter-language-pack==0.5.0`. If installation fails on your platform, install `tree_sitter_languages` instead — the `CodeSplitter` import is unchanged. The `chunking_service.py` has a fallback to `SentenceWindowNodeParser` if `CodeSplitter` fails, and will log a warning.

---

## Acceptance Checklist

- [x] Upload a PDF → reaches `status='completed'` asynchronously via Celery
- [x] Upload a `.py` file → reaches `status='completed'` asynchronously via Celery
- [x] `/query` returns chunks with cross-encoder rerank scores (not raw cosine)
- [x] `/query` against code file returns windowed code context
- [x] `document_id` filter narrows results to that document only
- [x] Delete removes only that document's vectors (sibling chunks untouched)
- [x] `embed_dim=384` set consistently everywhere (PGVectorStore, config.py, embedding calls)
- [x] No route handler contains raw SQL or direct PGVectorStore/HuggingFaceEmbedding calls
- [x] Tables managed by Alembic migrations (not auto-created at runtime)
- [x] README covers: architecture diagram, schema, chunking strategy, scaling strategy, trade-offs, setup instructions

---

## Project Structure

```
ActoWizRAG/
├── app/
│   ├── api/v1/
│   │   ├── documents.py        # upload, list, get, delete endpoints
│   │   └── query.py            # semantic query endpoint
│   ├── services/
│   │   ├── ingestion_service.py  # load → chunk → embed → store
│   │   ├── chunking_service.py   # sentence-window vs CodeSplitter strategy
│   │   ├── embedding_service.py  # HuggingFaceEmbedding wrapper (384-dim)
│   │   ├── retrieval_service.py  # retrieve → rerank → window-replace
│   │   ├── llm_service.py        # AI Gateway: Groq / OpenAI-compatible
│   │   └── rag_service.py        # coordinates retrieval + LLM
│   ├── repositories/
│   │   ├── vector_repository.py    # PGVectorStore insert/delete/search
│   │   └── document_repository.py  # SQLAlchemy CRUD for documents + query_logs
│   ├── models/
│   │   ├── orm.py       # SQLAlchemy models + async engine + session factory
│   │   ├── request.py   # Pydantic v2 request schemas
│   │   └── response.py  # Pydantic v2 response schemas
│   ├── workers/
│   │   ├── celery_app.py         # Celery factory (Redis broker + backend)
│   │   └── ingestion_tasks.py    # ingest_document_task, delete_document_task
│   ├── core/
│   │   ├── config.py             # pydantic-settings, .env loading
│   │   ├── logger.py             # structured JSON logging
│   │   └── exceptions.py         # custom exceptions + FastAPI handlers
│   ├── utils/
│   │   └── file_storage.py       # async file save, type detection
│   └── main.py                   # FastAPI app factory, lifespan, router mounting
├── alembic/
│   ├── env.py                    # Alembic configuration (reads DATABASE_SYNC_URL)
│   ├── script.py.mako            # Migration file template
│   └── versions/
│       └── 0001_initial_schema.py  # Initial migration: documents + query_logs
├── tests/
│   ├── unit/
│   │   ├── test_chunking_service.py
│   │   ├── test_embedding_service.py
│   │   └── test_retrieval_service.py
│   └── integration/
│       └── test_e2e_pipeline.py
├── scripts/
│   └── init_db.sql               # Docker postgres init: CREATE EXTENSION vector
├── alembic.ini                   # Alembic configuration file
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
