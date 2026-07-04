"""
Chunking service — selects the appropriate chunking strategy per file type.

Strategy:
  - Text / PDF / Markdown → SentenceWindowNodeParser
      Embeds individual sentences for precision; stores a ±window_size sentence
      window in metadata for context expansion at retrieval time.

  - Code (.py / .js / …) → CodeSplitter + manual window augmentation
      CodeSplitter produces tight code chunks (40 lines, 15-line overlap).
      We then augment each chunk's metadata with a "window" field built from
      the preceding + current + following chunk's raw text, mirroring the
      sentence-window idea: embed the tight chunk, but expose wider surrounding
      code as retrievable context.

Every node receives:
  - metadata["document_id"]  → links back to the `documents` table row
  - metadata["file_type"]    → aids metadata filtering at query time
  - metadata["window"]       → wide context for MetadataReplacementPostProcessor
  - metadata["original_text"]→ the original sentence (sentence-window path only)
"""

from __future__ import annotations

from typing import Sequence

from llama_index.core.schema import Document as LlamaDocument, TextNode, BaseNode
from llama_index.core.node_parser import SentenceWindowNodeParser, CodeSplitter

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class ChunkingService:
    """
    Parses raw document text into LlamaIndex nodes ready for embedding.

    Usage::

        service = ChunkingService()
        nodes = service.chunk(text, document_id=doc_id, file_type="pdf")
    """

    def __init__(self) -> None:
        self._sentence_parser = SentenceWindowNodeParser.from_defaults(
            window_size=settings.SENTENCE_WINDOW_SIZE,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        logger.info(
            "ChunkingService initialised",
            extra={"sentence_window_size": settings.SENTENCE_WINDOW_SIZE},
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def chunk(
        self,
        text: str,
        *,
        document_id: str,
        file_type: str,
        language: str | None = None,
    ) -> list[BaseNode]:
        """
        Chunk `text` using the strategy appropriate for `file_type`.

        Args:
            text:        Raw document text.
            document_id: UUID of the parent `documents` row.
            file_type:   "pdf" | "text" | "markdown" | "code"
            language:    Tree-sitter language name (required when file_type="code").

        Returns:
            List of LlamaIndex TextNode objects ready for embedding.
        """
        if file_type == "code":
            nodes = self._chunk_code(text, language=language or "python")
        else:
            nodes = self._chunk_prose(text)

        # Tag every node with shared metadata.
        for node in nodes:
            node.metadata["document_id"] = document_id
            node.metadata["file_type"] = file_type
            # Exclude large window/original_text from embedding to avoid noise.
            node.excluded_embed_metadata_keys = ["window", "original_text"]
            node.excluded_llm_metadata_keys = ["window"]

        logger.info(
            "Chunking complete",
            extra={
                "document_id": document_id,
                "file_type": file_type,
                "node_count": len(nodes),
            },
        )
        return nodes

    # ── Private helpers ────────────────────────────────────────────────────────

    def _chunk_prose(self, text: str) -> list[BaseNode]:
        """
        Apply SentenceWindowNodeParser to prose (PDF / text / markdown).

        Each node's text = one sentence.
        Each node's metadata["window"] = surrounding ±window_size sentences.
        """
        llama_doc = LlamaDocument(text=text)
        nodes = self._sentence_parser.get_nodes_from_documents([llama_doc])
        return nodes

    def _chunk_code(self, text: str, language: str) -> list[BaseNode]:
        """
        Apply CodeSplitter to code files, then augment each chunk with a
        "window" field containing the preceding + current + following chunk text.
        """
        # CodeSplitter may raise if tree-sitter grammar is unavailable;
        # we fall back to prose chunking in that case.
        try:
            splitter = CodeSplitter(
                language=language,
                chunk_lines=settings.CODE_CHUNK_LINES,
                chunk_lines_overlap=settings.CODE_CHUNK_OVERLAP,
                max_chars=settings.CODE_CHUNK_MAX_CHARS,
            )
            llama_doc = LlamaDocument(text=text)
            nodes: list[BaseNode] = splitter.get_nodes_from_documents([llama_doc])
            logger.info(
                "Code split complete",
                extra={"language": language, "node_count": len(nodes)},
            )
        except Exception as exc:
            logger.warning(
                "CodeSplitter failed; falling back to sentence-window",
                extra={"language": language, "error": str(exc)},
            )
            return self._chunk_prose(text)

        # Window augmentation — mirror sentence-window for code.
        texts = [n.get_content() for n in nodes]
        for i, node in enumerate(nodes):
            prev_text = texts[i - 1] if i > 0 else ""
            curr_text = texts[i]
            next_text = texts[i + 1] if i < len(texts) - 1 else ""
            node.metadata["window"] = "\n\n".join(
                filter(None, [prev_text, curr_text, next_text])
            )
            node.metadata["original_text"] = curr_text

        return nodes
