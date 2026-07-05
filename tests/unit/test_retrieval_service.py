"""
Unit tests for RetrievalService.

Uses mocked VectorRepository and EmbeddingService — no real DB or models needed.
"""

import pytest
from unittest.mock import MagicMock
from llama_index.core.schema import TextNode, NodeWithScore


def _make_node(node_id: str, text: str, doc_id: str = "doc-001", score: float = 0.5) -> NodeWithScore:
    node = TextNode(
        text=text,
        id_=node_id,
        metadata={
            "document_id": doc_id,
            "source_document_id": doc_id,
            "window": f"WINDOW: {text}",
            "file_type": "text",
        },
    )
    return NodeWithScore(node=node, score=score)


@pytest.fixture
def mock_vector_repo():
    repo = MagicMock()
    repo.search.return_value = [
        _make_node("n1", "Python is a programming language", score=0.95),
        _make_node("n2", "The sky is blue", score=0.80),
        _make_node("n3", "Machine learning models need training data", score=0.75),
    ]
    return repo


@pytest.fixture
def mock_embedding_svc():
    svc = MagicMock()
    svc.embed_query.return_value = [0.1] * 384
    return svc


class TestRetrievalService:
    def test_retrieve_returns_results(self, mock_vector_repo, mock_embedding_svc):
        """Basic smoke test — retrieve returns a list of ChunkResult."""
        from app.services.retrieval_service import RetrievalService

        svc = RetrievalService(
            vector_repo=mock_vector_repo,
            embedding_service=mock_embedding_svc,
        )
        results = svc.retrieve("What is Python?", top_k=2)

        assert len(results) == 2
        assert results[0].node_id == "n1"

    def test_results_ordered_by_cosine_score(self, mock_vector_repo, mock_embedding_svc):
        """
        Without a reranker, results should follow the cosine similarity order
        returned by the vector store (n1 > n2 > n3).
        """
        from app.services.retrieval_service import RetrievalService

        svc = RetrievalService(
            vector_repo=mock_vector_repo,
            embedding_service=mock_embedding_svc,
        )
        results = svc.retrieve("machine learning", top_k=3)

        assert results[0].node_id == "n1"
        assert results[0].score >= results[1].score >= results[2].score

    def test_empty_candidates_returns_empty_list(self, mock_embedding_svc):
        """If vector search returns nothing, retrieve() returns an empty list."""
        empty_repo = MagicMock()
        empty_repo.search.return_value = []

        from app.services.retrieval_service import RetrievalService

        svc = RetrievalService(
            vector_repo=empty_repo,
            embedding_service=mock_embedding_svc,
        )
        results = svc.retrieve("anything", top_k=5)

        assert results == []

    def test_window_text_replaces_node_content(self, mock_vector_repo, mock_embedding_svc):
        """
        Window expansion should replace node content with the wider window text
        before mapping to ChunkResult.
        """
        from app.services.retrieval_service import RetrievalService

        svc = RetrievalService(
            vector_repo=mock_vector_repo,
            embedding_service=mock_embedding_svc,
        )
        results = svc.retrieve("Python", top_k=1)

        assert len(results) == 1
        # Content should contain the window prefix added in _make_node()
        assert "WINDOW:" in results[0].content

    def test_results_respect_top_k(self, mock_vector_repo, mock_embedding_svc):
        """retrieve() should return at most top_k results."""
        from app.services.retrieval_service import RetrievalService

        svc = RetrievalService(
            vector_repo=mock_vector_repo,
            embedding_service=mock_embedding_svc,
        )
        results = svc.retrieve("anything", top_k=1)
        assert len(results) == 1

    def test_document_id_propagated_to_chunk_result(self, mock_vector_repo, mock_embedding_svc):
        """document_id from node metadata must be set on the ChunkResult."""
        from app.services.retrieval_service import RetrievalService

        svc = RetrievalService(
            vector_repo=mock_vector_repo,
            embedding_service=mock_embedding_svc,
        )
        results = svc.retrieve("Python", top_k=1)

        assert results[0].document_id == "doc-001"


class TestDeletePartialFailure:
    """Tests for the partial-failure path in IngestionService.delete()."""

    def test_fallback_metadata_delete_called_on_primary_failure(self):
        """If delete_by_document_id fails, delete_by_document_id_metadata must be called."""
        mock_repo = MagicMock()
        mock_repo.delete_by_document_id.side_effect = Exception("pgvector error")

        from app.services.ingestion_service import IngestionService

        svc = IngestionService(
            vector_repo=mock_repo,
            chunking_service=MagicMock(),
            embedding_service=MagicMock(),
        )
        svc.delete("doc-partial-fail")

        mock_repo.delete_by_document_id.assert_called_once_with("doc-partial-fail")
        mock_repo.delete_by_document_id_metadata.assert_called_once_with("doc-partial-fail")

    def test_no_fallback_when_primary_succeeds(self):
        """If primary deletion succeeds, fallback should NOT be called."""
        mock_repo = MagicMock()
        mock_repo.delete_by_document_id.return_value = None  # success

        from app.services.ingestion_service import IngestionService

        svc = IngestionService(
            vector_repo=mock_repo,
            chunking_service=MagicMock(),
            embedding_service=MagicMock(),
        )
        svc.delete("doc-clean-delete")

        mock_repo.delete_by_document_id.assert_called_once()
        mock_repo.delete_by_document_id_metadata.assert_not_called()
