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

    # ── Sentence-window chunking ───────────────────────────────────────────────
    SENTENCE_WINDOW_SIZE: int = Field(
        default=3,
        description="Number of surrounding sentences to store as window context",
    )

    # ── Code chunking ──────────────────────────────────────────────────────────
    CODE_CHUNK_LINES: int = Field(default=40)
    CODE_CHUNK_OVERLAP: int = Field(default=15)
    CODE_CHUNK_MAX_CHARS: int = Field(default=1500)

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
        default="llama3-8b-8192",
        description="Model name / deployment name for the active LLM provider",
    )
    LLM_MAX_TOKENS: int = Field(default=1024)
    LLM_TEMPERATURE: float = Field(default=0.1)

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
