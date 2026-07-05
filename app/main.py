"""
FastAPI application factory.

Responsibilities:
  - Create the FastAPI app with metadata.
  - Register exception handlers.
  - Register API routers.
  - Run DB initialisation on startup.
  - Configure structured logging.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logger import configure_root_logger, get_logger
from app.models.orm import engine, verify_db_connection
from app.models.response import HealthResponse
from app.api.v1.documents import router as documents_router
from app.api.v1.query import router as query_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle startup and shutdown events.

    Tables are NOT created here — they are managed by Alembic migrations.
    Run `alembic upgrade head` before starting the API for the first time.
    """
    configure_root_logger(settings.LOG_LEVEL)
    logger.info("Starting ActoWiz RAG API", extra={"env": settings.APP_ENV})
    await verify_db_connection()

    # Preload embedding + reranker models so the first /query in Swagger is fast.
    import asyncio
    from app.api.v1.query import get_rag_service

    loop = asyncio.get_running_loop()
    logger.info("Preloading RAG models (first query may take ~30s on cold start)...")
    await loop.run_in_executor(None, get_rag_service)
    logger.info("RAG models ready")

    yield
    logger.info("Shutting down ActoWiz RAG API")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ActoWiz Internal AI Knowledge Platform",
        description=(
            "Production-grade RAG backend for internal developer knowledge search. "
            "Supports PDF, DOCX, Markdown, plain-text, and code ingestion with "
            "sentence-window retrieval, cross-encoder reranking, and optional "
            "LLM-powered answer synthesis."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten in production to specific origins.
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ─────────────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(documents_router, prefix=settings.API_V1_PREFIX)
    app.include_router(query_router, prefix=settings.API_V1_PREFIX)

    # ── Health check ───────────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health() -> HealthResponse:
        import sqlalchemy as sa

        from app.core.config import settings
        from app.repositories.vector_repository import PHYSICAL_CHUNK_TABLE_NAME

        db_status = "ok"
        chunk_count: int | None = None

        try:
            async with engine.begin() as conn:
                await conn.execute(sa.text("SELECT 1"))
                table_exists = await conn.execute(
                    sa.text(
                        "SELECT EXISTS (SELECT FROM information_schema.tables "
                        "WHERE table_schema='public' AND table_name=:t)"
                    ),
                    {"t": PHYSICAL_CHUNK_TABLE_NAME},
                )
                if table_exists.scalar():
                    row = await conn.execute(
                        sa.text(f"SELECT COUNT(*) FROM {PHYSICAL_CHUNK_TABLE_NAME}")
                    )
                    chunk_count = int(row.scalar() or 0)
        except Exception as exc:
            db_status = f"error: {exc}"
            logger.warning("Health check DB probe failed", extra={"error": str(exc)})

        redis_status = "ok"
        try:
            import redis

            client = redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
            client.ping()
        except Exception as exc:
            redis_status = f"error: {exc}"

        llm_ok = bool(
            settings.LLM_PROVIDER != "none" and settings.LLM_API_KEY
        )

        overall = "ok" if db_status == "ok" and redis_status == "ok" else "degraded"

        return HealthResponse(
            status=overall,
            database=db_status,
            redis=redis_status,
            stored_chunks=chunk_count,
            llm_configured=llm_ok,
        )

    return app


app = create_app()
