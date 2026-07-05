"""
Embedding service — wraps HuggingFaceEmbedding.

A single instance is created at startup and reused throughout the application
(both in the FastAPI process for query embedding and in the Celery worker for
ingestion embedding).  This avoids repeated model loading overhead.

embed_dim=384 is the output dimension of "sentence-transformers/all-MiniLM-L6-v2".
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Thin wrapper around HuggingFaceEmbedding.

    Ensures the same model instance and configuration are used for both
    ingestion (Celery worker) and retrieval (FastAPI handler).
    """

    def __init__(self) -> None:
        logger.info(
            "Loading embedding model",
            extra={"model": settings.EMBED_MODEL_NAME, "embed_dim": settings.EMBED_DIM},
        )
        self._model = HuggingFaceEmbedding(
            model_name=settings.EMBED_MODEL_NAME,
            embed_batch_size=32,
            # max_length defaults to the model's tokenizer max — left as-is.
        )
        logger.info("Embedding model loaded successfully")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        embedding = self._model.get_query_embedding(text)
        if len(embedding) != settings.EMBED_DIM:
            raise ValueError(
                f"Query embedding dim {len(embedding)} != EMBED_DIM {settings.EMBED_DIM}. "
                "Check EMBED_MODEL_NAME and EMBED_DIM in .env."
            )
        return embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed a list of texts (used during ingestion)."""
        embeddings = self._model.get_text_embedding_batch(texts, show_progress=True)
        for embedding in embeddings:
            if len(embedding) != settings.EMBED_DIM:
                raise ValueError(
                    f"Document embedding dim {len(embedding)} != EMBED_DIM {settings.EMBED_DIM}. "
                    "Check EMBED_MODEL_NAME and EMBED_DIM in .env."
                )
        return embeddings

    def get_model(self) -> HuggingFaceEmbedding:
        """Return the underlying LlamaIndex embedding model (for node embedding)."""
        return self._model
