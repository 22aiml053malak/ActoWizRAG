"""
Unit tests for RetrievalService.

Key verification: reranking CHANGES the ordering compared to raw cosine similarity.
Uses mocked VectorRepository, EmbeddingService, and SentenceTransformerRerank.
"""

import pytest
from unittest.mock import MagicMock, patch
from llama_index.core.schema import TextNode, NodeWithScore


def _make_node(node_id: str, text: str, doc_id: str = "doc-001", score: float = 0.5) -> NodeWithScore:
    node = TextNode(
        text=text,
        id_=node_id,
        metadata={"document_id": doc_id, "window": f"WINDOW: {text}", "file_type": "text"},
    )
    return NodeWithScore(node=node, score=score)


@pytest.fixture
def mock_vector_repo():
    repo = MagicMock()
    # Return 3 candidates with decreasing cosine similarity scores.
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
        # Mock the reranker to pass through nodes unchanged (reversed for test).
        with patch(
            "app.services.retrieval_service.SentenceTransformerRerank"
        ) as mock_rerank_cls, patch(
            "app.services.retrieval_service.MetadataReplacementPostProcessor"
        ) as mock_window_cls:
            mock_reranker = MagicMock()
            mock_reranker.postprocess_nodes.return_value = mock_vector_repo.search.return_value[:2]
            mock_rerank_cls.return_value = mock_reranker

            mock_window = MagicMock()
            mock_window.postprocess_nodes.return_value = mock_vector_repo.search.return_value[:2]
            mock_window_cls.return_value = mock_window

            from app.services.retrieval_service import RetrievalService
            svc = RetrievalService(
                vector_repo=mock_vector_repo,
                embedding_service=mock_embedding_svc,
            )
            results = svc.retrieve("What is Python?", top_k=2)

        assert len(results) == 2
        assert results[0].node_id == "n1"

    def test_reranking_changes_ordering(self, mock_vector_repo, mock_embedding_svc):
        """
        Verify that reranking CAN produce a different order than raw cosine similarity.

        The vector store returns: n1 (0.95) > n2 (0.80) > n3 (0.75)
        We mock the reranker to return them reversed: n3 > n2 > n1
        The test asserts the final result follows the reranker's order, not the cosine order.
        """
        raw_candidates = mock_vector_repo.search.return_value

        reranked_order = [
            _make_node("n3", "Machine learning models need training data", score=0.99),
            _make_node("n2", "The sky is blue", score=0.60),
            _make_node("n1", "Python is a programming language", score=0.20),
        ]

        with patch(
            "app.services.retrieval_service.SentenceTransformerRerank"
        ) as mock_rerank_cls, patch(
            "app.services.retrieval_service.MetadataReplacementPostProcessor"
        ) as mock_window_cls:
            mock_reranker = MagicMock()
            mock_reranker.postprocess_nodes.return_value = reranked_order
            mock_rerank_cls.return_value = mock_reranker

            mock_window = MagicMock()
            mock_window.postprocess_nodes.return_value = reranked_order
            mock_window_cls.return_value = mock_window

            from app.services.retrieval_service import RetrievalService
            svc = RetrievalService(
                vector_repo=mock_vector_repo,
                embedding_service=mock_embedding_svc,
            )
            results = svc.retrieve("machine learning", top_k=3)

        # After reranking, n3 should be first (not n1 as per cosine similarity).
        assert results[0].node_id == "n3", (
            "Reranking should have changed ordering from cosine similarity order"
        )
        assert results[0].score > results[2].score

    def test_empty_candidates_returns_empty_list(self, mock_embedding_svc):
        """If vector search returns nothing, retrieve() returns an empty list."""
        empty_repo = MagicMock()
        empty_repo.search.return_value = []

        with patch("app.services.retrieval_service.SentenceTransformerRerank"), \
             patch("app.services.retrieval_service.MetadataReplacementPostProcessor"):
            from app.services.retrieval_service import RetrievalService
            svc = RetrievalService(
                vector_repo=empty_repo,
                embedding_service=mock_embedding_svc,
            )
            results = svc.retrieve("anything", top_k=5)

        assert results == []

    def test_metadata_filter_passed_to_vector_search(self, mock_vector_repo, mock_embedding_svc):
        """document_id filter must be forwarded to the vector repository."""
        with patch("app.services.retrieval_service.SentenceTransformerRerank") as mock_rerank_cls, \
             patch("app.services.retrieval_service.MetadataReplacementPostProcessor") as mock_window_cls:
            mock_reranker = MagicMock()
            mock_reranker.postprocess_nodes.return_value = mock_vector_repo.search.return_value[:1]
            mock_rerank_cls.return_value = mock_reranker
            mock_window = MagicMock()
            mock_window.postprocess_nodes.return_value = mock_vector_repo.search.return_value[:1]
            mock_window_cls.return_value = mock_window

            from app.services.retrieval_service import RetrievalService
            svc = RetrievalService(
                vector_repo=mock_vector_repo,
                embedding_service=mock_embedding_svc,
            )
            svc.retrieve("query", top_k=1, document_id="specific-doc-uuid")

        mock_vector_repo.search.assert_called_once()
        call_kwargs = mock_vector_repo.search.call_args.kwargs
        assert call_kwargs["document_id"] == "specific-doc-uuid"


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
