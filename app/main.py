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
from app.models.orm import verify_db_connection
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
    yield
    logger.info("Shutting down ActoWiz RAG API")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ActoWiz Internal AI Knowledge Platform",
        description=(
            "Production-grade RAG backend for internal developer knowledge search. "
            "Supports PDF, Markdown, plain-text, and code ingestion with "
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
        return HealthResponse()

    return app


app = create_app()
