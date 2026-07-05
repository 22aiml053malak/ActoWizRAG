"""
Core configuration module.
Uses pydantic-settings to load configuration from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Literal
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ───────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/actowiz_rag",
        description="Async PostgreSQL connection string",
    )
    DATABASE_SYNC_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/actowiz_rag",
        description="Sync PostgreSQL connection string (used by PGVectorStore / alembic)",
    )

    # ── Redis / Celery ─────────────────────────────────────────────────────────
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string (broker + result backend)",
    )

    # ── Embedding ──────────────────────────────────────────────────────────────
    EMBED_MODEL_NAME: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model name",
    )
    EMBED_DIM: int = Field(
        default=384,
        description="Embedding dimensionality — must match the model and PGVectorStore embed_dim",
    )

    # ── Reranker ───────────────────────────────────────────────────────────────
    RERANK_MODEL_NAME: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder reranker model name",
    )
    RERANK_TOP_N: int = Field(
        default=5,
        description="Number of candidates to keep after reranking",
    )

    # ── Retrieval ──────────────────────────────────────────────────────────────
    SIMILARITY_TOP_K: int = Field(
        default=15,
        description="Wide candidate set size for initial vector search",
    )

    # ── Structure-aware chunking ───────────────────────────────────────────────
    STRUCTURE_CHUNK_MAX_CHARS: int = Field(
        default=1200,
        description="Target chunk size for structure-aware prose chunking",
    )
    PAGE_CHUNK_OVERLAP_CHARS: int = Field(
        default=200,
        description="Character overlap between adjacent page chunks for sliding-window context",
    )

    # ── Code chunking ──────────────────────────────────────────────────────────
    CODE_CHUNK_LINES: int = Field(default=40)
    CODE_CHUNK_OVERLAP: int = Field(default=15)
    CODE_CHUNK_MAX_CHARS: int = Field(default=1500)

    # ── PDF / OCR loading ────────────────────────────────────────────────────
    PDF_NATIVE_TEXT_MIN_CHARS: int = Field(
        default=300,
        description="Minimum extracted PDF text length before OCR fallback is skipped",
    )
    PADDLEOCR_LANG: str = Field(
        default="en",
        description="PaddleOCR language code (en, ch, fr, es, etc.)",
    )
    PADDLEOCR_DPI: int = Field(
        default=200,
        description="DPI used when rendering PDF pages for PaddleOCR structure extraction",
    )
    OCR_DPI: int = Field(
        default=300,
        description="DPI used when rendering scanned PDFs for tesseract OCR (last-resort fallback)",
    )

    # ── Storage ────────────────────────────────────────────────────────────────
    STORAGE_DIR: str = Field(
        default="./storage/uploads",
        description="Directory where uploaded files are persisted on disk",
    )

    # ── LLM / AI Gateway ──────────────────────────────────────────────────────
    LLM_PROVIDER: Literal["groq", "openai_compatible", "none"] = Field(
        default="none",
        description="Active LLM provider. Set 'none' to disable answer generation.",
    )
    LLM_API_KEY: str = Field(default="", description="API key for the active LLM provider")
    LLM_API_BASE_URL: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Base URL for OpenAI-compatible providers",
    )
    LLM_MODEL_NAME: str = Field(
        default="llama-3.1-8b-instant",
        description="Model name / deployment name for the active LLM provider",
    )
    LLM_MAX_TOKENS: int = Field(default=1024)
    LLM_TEMPERATURE: float = Field(default=0.1)

    # ── Vision Document Parsing ───────────────────────────────────────────────
    # Deprecated — replaced by PaddleOCR. Kept for backward compatibility with
    # existing .env files but no longer used by document_loader_service.py.
    VISION_MODEL_NAME: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        description="[DEPRECATED] Vision-capable model used for OCR/layout understanding.",
    )
    VISION_RENDER_DPI: int = Field(
        default=150,
        description="[DEPRECATED] DPI used when rendering PDF pages before sending to the vision model.",
    )
    VISION_MAX_TOKENS: int = Field(
        default=4096,
        description="[DEPRECATED] Maximum output tokens returned by the vision model.",
    )
    # ── App ────────────────────────────────────────────────────────────────────
    APP_ENV: Literal["development", "staging", "production"] = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")
    API_V1_PREFIX: str = Field(default="/api/v1")

    @field_validator("STORAGE_DIR")
    @classmethod
    def ensure_storage_dir(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v


# Singleton instance — import `settings` everywhere
settings = Settings()
