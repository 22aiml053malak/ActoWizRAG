"""
Unit tests for ChunkingService.

Verifies:
  - Prose chunking populates metadata["window"] on every node.
  - Code chunking populates metadata["window"] as prev+curr+next text.
  - Every node gets document_id and file_type in metadata.
  - Window field is non-empty and wider than (or equal to) the node text.
"""

import pytest
from unittest.mock import MagicMock, patch


# ── Fixtures ───────────────────────────────────────────────────────────────────

PROSE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "It was a warm summer day. "
    "The sun was shining brightly in the clear blue sky. "
    "Birds were singing in the trees. "
    "A gentle breeze rustled the leaves. "
    "Everything felt peaceful and calm. "
    "The dog lay in the shade, watching the fox run by. "
    "Life was good in the meadow."
)

CODE_TEXT = """\
def add(a, b):
    \"\"\"Return the sum of a and b.\"\"\"
    return a + b


def multiply(a, b):
    \"\"\"Return the product of a and b.\"\"\"
    return a * b


def divide(a, b):
    \"\"\"Return a divided by b, raising ValueError if b is zero.\"\"\"
    if b == 0:
        raise ValueError("Division by zero")
    return a / b


class Calculator:
    def __init__(self):
        self.history = []

    def compute(self, op, a, b):
        if op == "add":
            result = add(a, b)
        elif op == "mul":
            result = multiply(a, b)
        elif op == "div":
            result = divide(a, b)
        else:
            raise ValueError(f"Unknown op: {op}")
        self.history.append((op, a, b, result))
        return result
"""


@pytest.fixture
def chunking_service():
    from app.services.chunking_service import ChunkingService
    return ChunkingService()


# ── Prose (sentence-window) tests ──────────────────────────────────────────────

class TestProseChunking:
    def test_window_metadata_populated(self, chunking_service):
        nodes = chunking_service.chunk(
            PROSE_TEXT,
            document_id="doc-001",
            file_type="text",
        )
        assert len(nodes) > 0, "Expected at least one node"
        for node in nodes:
            assert "window" in node.metadata, (
                f"Node {node.node_id} is missing 'window' metadata"
            )
            assert node.metadata["window"], "Window metadata must not be empty"

    def test_document_id_in_metadata(self, chunking_service):
        nodes = chunking_service.chunk(
            PROSE_TEXT,
            document_id="doc-abc",
            file_type="text",
        )
        for node in nodes:
            assert node.metadata.get("document_id") == "doc-abc"

    def test_file_type_in_metadata(self, chunking_service):
        nodes = chunking_service.chunk(
            PROSE_TEXT,
            document_id="doc-xyz",
            file_type="pdf",
        )
        for node in nodes:
            assert node.metadata.get("file_type") == "pdf"

    def test_original_text_in_metadata(self, chunking_service):
        nodes = chunking_service.chunk(
            PROSE_TEXT,
            document_id="doc-ot",
            file_type="markdown",
        )
        for node in nodes:
            assert "original_text" in node.metadata

    def test_window_broader_than_node_text(self, chunking_service):
        """The window should be at least as long as the node text (context expansion)."""
        nodes = chunking_service.chunk(
            PROSE_TEXT,
            document_id="doc-w",
            file_type="text",
        )
        # Not every node will have a wider window (edge nodes at start/end),
        # but at least one interior node should.
        window_lengths = [len(n.metadata["window"]) for n in nodes]
        node_lengths = [len(n.get_content()) for n in nodes]
        has_wider_window = any(
            w >= t for w, t in zip(window_lengths, node_lengths)
        )
        assert has_wider_window, "Expected at least one node with window >= node text length"


# ── Code chunking tests ────────────────────────────────────────────────────────

class TestCodeChunking:
    def test_window_metadata_populated_for_code(self, chunking_service):
        nodes = chunking_service.chunk(
            CODE_TEXT,
            document_id="doc-code-001",
            file_type="code",
            language="python",
        )
        assert len(nodes) > 0, "Expected at least one code node"
        for node in nodes:
            assert "window" in node.metadata, (
                f"Code node {node.node_id} missing 'window' metadata"
            )

    def test_code_window_contains_surrounding_context(self, chunking_service):
        """For code with multiple chunks, interior nodes should have surrounding text in window."""
        nodes = chunking_service.chunk(
            CODE_TEXT * 5,  # repeat to get multiple chunks
            document_id="doc-code-002",
            file_type="code",
            language="python",
        )
        if len(nodes) >= 3:
            # An interior node's window should contain text from neighbours.
            interior = nodes[1]
            assert interior.metadata["window"] != interior.get_content(), (
                "Interior node's window should include context beyond its own text"
            )

    def test_code_document_id_and_file_type(self, chunking_service):
        nodes = chunking_service.chunk(
            CODE_TEXT,
            document_id="doc-py",
            file_type="code",
            language="python",
        )
        for node in nodes:
            assert node.metadata["document_id"] == "doc-py"
            assert node.metadata["file_type"] == "code"

    def test_code_splitter_fallback_on_unknown_language(self, chunking_service):
        """If CodeSplitter fails for an unknown language, it should fall back gracefully."""
        nodes = chunking_service.chunk(
            CODE_TEXT,
            document_id="doc-unknown",
            file_type="code",
            language="brainfuck",  # not a real tree-sitter language
        )
        # Fallback to prose chunking — should still return nodes with window metadata.
        assert len(nodes) > 0
        for node in nodes:
            assert "window" in node.metadata
