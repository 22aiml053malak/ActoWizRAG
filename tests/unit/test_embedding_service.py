"""
Unit tests for EmbeddingService.

Uses a mock HuggingFaceEmbedding to avoid loading the real model in fast unit tests.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_hf_embedding():
    """Return a mock that mimics HuggingFaceEmbedding."""
    mock = MagicMock()
    mock.get_query_embedding.return_value = [0.1] * 384
    mock.get_text_embedding_batch.return_value = [[0.1] * 384, [0.2] * 384]
    return mock


class TestEmbeddingService:
    def test_embed_query_returns_384_dim_vector(self, mock_hf_embedding):
        with patch(
            "app.services.embedding_service.HuggingFaceEmbedding",
            return_value=mock_hf_embedding,
        ):
            from app.services.embedding_service import EmbeddingService
            svc = EmbeddingService()
            result = svc.embed_query("What is machine learning?")

        assert isinstance(result, list)
        assert len(result) == 384

    def test_embed_texts_returns_correct_count(self, mock_hf_embedding):
        with patch(
            "app.services.embedding_service.HuggingFaceEmbedding",
            return_value=mock_hf_embedding,
        ):
            from app.services.embedding_service import EmbeddingService
            svc = EmbeddingService()
            results = svc.embed_texts(["text one", "text two"])

        assert len(results) == 2
        assert all(len(v) == 384 for v in results)

    def test_get_model_returns_underlying_model(self, mock_hf_embedding):
        with patch(
            "app.services.embedding_service.HuggingFaceEmbedding",
            return_value=mock_hf_embedding,
        ):
            from app.services.embedding_service import EmbeddingService
            svc = EmbeddingService()
            assert svc.get_model() is mock_hf_embedding
