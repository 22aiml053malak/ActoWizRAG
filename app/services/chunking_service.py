"""
Chunking service — selects the appropriate chunking strategy per file type.

Strategy:
  - Text / PDF / DOCX / Markdown → page-wise, structure-aware Markdown chunking
      The old strategy here was SentenceWindowNodeParser: embed one sentence
      at a time, store a ±N-sentence window in metadata for context expansion
      at retrieval time. That's a poor fit for documents with headings,
      tables, and bullet lists (reports, whitepapers, decks) — a sentence
      splitter has no notion of "this is a table row" or "this heading
      introduces the next three paragraphs," so it shreds tables into
      disconnected fragments and disassociates headings from their content.

      This version parses the document into structural blocks — headings,
      paragraphs, Markdown tables, bullet/numbered lists, blockquotes — and
      groups adjacent blocks into chunks, with three hard rules, in priority
      order: (1) a table is always its own chunk, never split or merged
      with surrounding prose; (2) a page boundary always starts a new
      chunk — this is what makes chunking genuinely page-wise, so a chunk
      never silently spans two source pages; and (3) a heading always
      starts a new chunk, so a chunk reads as "heading + the content under
      it" rather than an arbitrary character window. A single page whose
      content exceeds STRUCTURE_CHUNK_MAX_CHARS is still split further,
      using a sliding character window (see below) so no single chunk
      grows unbounded.

      Each chunk carries `page` metadata recovered from `## Page N` markers
      (already emitted by document_loader_service.py for every extraction
      tier — native, PaddleOCR, and tesseract alike) and `section` metadata
      from the nearest heading above it, enabling page-level citations and
      metadata filtering at query time.

      Sliding-window overlap: metadata["window"] is built from the last
      PAGE_CHUNK_OVERLAP_CHARS characters of the previous chunk + the full
      current chunk + the first PAGE_CHUNK_OVERLAP_CHARS characters of the
      next chunk — a trimmed overlap rather than pulling in a whole
      neighboring chunk. retrieval_service.py's MetadataReplacementPostProcessor
      swaps each chunk's embedded text for this window at query time, so
      retrieval gets the tight, page-scoped chunk (good for precision) while
      the LLM/answer-generation step gets the overlap-expanded context (good
      for not losing content that sits right at a page/chunk boundary).

  - Code (.py / .js / …) → CodeSplitter + manual window augmentation
      CodeSplitter produces tight code chunks (40 lines, 15-line overlap).
      We then augment each chunk's metadata with a "window" field built from
      the preceding + current + following chunk's raw text, mirroring the
      same context-expansion idea used for prose.

Every node receives:
  - metadata["document_id"]  → links back to the `documents` table row
  - metadata["source_document_id"] → non-reserved vector-store filter key
  - metadata["file_type"]    → aids metadata filtering at query time
  - metadata["window"]       → sliding-window context for MetadataReplacementPostProcessor
  - metadata["original_text"]→ the chunk's own raw text (prose + code paths)
  - metadata["page"]         → source page number, when recoverable (prose only)
  - metadata["section"]      → nearest heading above this chunk (prose only)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.node_parser import CodeSplitter

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# Target chunk size for structure-aware prose chunking. Falls back to a
# sane default if this hasn't been added to config.py yet.
STRUCTURE_CHUNK_MAX_CHARS = getattr(settings, "STRUCTURE_CHUNK_MAX_CHARS", 1200)

# Sliding-window overlap (chars) — used both (a) as the amount of the
# previous/next chunk pulled into metadata["window"], and (b) as the
# overlap amount when a single page must be split into multiple
# sub-chunks because it exceeds STRUCTURE_CHUNK_MAX_CHARS on its own.
PAGE_CHUNK_OVERLAP_CHARS = getattr(settings, "PAGE_CHUNK_OVERLAP_CHARS", 200)

_PAGE_MARKER_RE = re.compile(r"^#{1,6}\s*Page\s+(\d+)\s*$", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|")
_LIST_ITEM_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
_QUOTE_RE = re.compile(r"^\s*>")


@dataclass
class _Block:
    """One structural unit of a parsed document."""

    type: str  # "heading" | "paragraph" | "table" | "list" | "quote"
    text: str
    page: int | None
    section: str | None
    level: int | None = None  # heading level, only set when type == "heading"

    def render(self) -> str:
        if self.type == "heading":
            return f"{'#' * (self.level or 2)} {self.text}"
        return self.text


class ChunkingService:
    """
    Parses raw document text into LlamaIndex nodes ready for embedding.

    Usage::

        service = ChunkingService()
        nodes = service.chunk(text, document_id=doc_id, file_type="pdf")
    """

    def __init__(self) -> None:
        logger.info(
            "ChunkingService initialised",
            extra={
                "structure_chunk_max_chars": STRUCTURE_CHUNK_MAX_CHARS,
                "page_chunk_overlap_chars": PAGE_CHUNK_OVERLAP_CHARS,
            },
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
            file_type:   "pdf" | "docx" | "text" | "markdown" | "code"
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
            # LlamaIndex/PGVectorStore uses `document_id` internally while
            # serializing rows, so keep a separate stable key for filters.
            node.metadata["source_document_id"] = document_id
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

    def _simple_chunk_fallback(self, text: str) -> list[BaseNode]:
        """
        Fallback chunker when structure-aware parsing produces nothing usable.

        Splits text into fixed-size chunks with a sliding-window overlap.
        This handles edge cases like a document with no Markdown structure
        at all (plain unformatted text, no page markers).
        """
        chunk_size = 1000  # characters
        overlap = PAGE_CHUNK_OVERLAP_CHARS

        if not text or not text.strip():
            return []

        nodes: list[BaseNode] = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence/paragraph boundary if possible.
            if end < len(text):
                last_period = chunk_text.rfind(". ")
                last_newline = chunk_text.rfind("\n\n")
                break_point = max(last_period, last_newline)

                if break_point > chunk_size - 200:  # Only break if reasonably close to end
                    chunk_text = chunk_text[: break_point + 1]
                    end = start + break_point + 1

            chunk_text = chunk_text.strip()
            if chunk_text:
                window_start = max(0, start - overlap)
                window_end = min(len(text), end + overlap)
                window_text = text[window_start:window_end].strip()

                node = TextNode(
                    text=chunk_text,
                    metadata={
                        "window": window_text,
                        "original_text": chunk_text,
                    },
                )
                nodes.append(node)

            start = end - overlap if end < len(text) else end

        logger.info(
            "Simple fallback chunking complete",
            extra={"node_count": len(nodes)},
        )
        return nodes

    # ── Private helpers — prose ──────────────────────────────────────────────

    def _chunk_prose(self, text: str) -> list[BaseNode]:
        """
        Parse `text` into structural blocks (headings/paragraphs/tables/
        lists/quotes), group them page-wise into chunks, and build nodes
        with page/section metadata and sliding-window overlap.
        """
        try:
            blocks = self._parse_blocks(text)
            if not blocks:
                logger.warning("No structural blocks parsed, falling back to simple chunking")
                return self._simple_chunk_fallback(text)

            blocks = self._split_oversized_blocks(blocks)

            groups = self._group_blocks(blocks)
            if not groups:
                logger.warning("Block grouping produced nothing, falling back to simple chunking")
                return self._simple_chunk_fallback(text)

            return self._build_prose_nodes(groups)
        except Exception as exc:
            logger.warning(
                "Structure-aware chunking failed, falling back to simple chunking",
                extra={"error": str(exc)},
            )
            return self._simple_chunk_fallback(text)

    @staticmethod
    def _parse_blocks(text: str) -> list[_Block]:
        """
        Walk the document line by line, grouping contiguous lines of the
        same kind (table rows, list items, quote lines, paragraph lines)
        into single blocks. Headings and `## Page N` markers are handled
        specially: a heading becomes its own block and updates the
        "current section" tracker; a page marker updates the "current
        page" tracker without becoming a block itself.
        """
        blocks: list[_Block] = []
        current_page: int | None = None
        current_section: str | None = None

        buffer: list[str] = []
        buffer_type: str | None = None

        def flush() -> None:
            nonlocal buffer, buffer_type
            if buffer:
                content = "\n".join(buffer).strip()
                if content:
                    blocks.append(
                        _Block(
                            type=buffer_type or "paragraph",
                            text=content,
                            page=current_page,
                            section=current_section,
                        )
                    )
            buffer = []
            buffer_type = None

        for raw_line in text.split("\n"):
            stripped = raw_line.strip()

            if not stripped:
                flush()
                continue

            page_match = _PAGE_MARKER_RE.match(stripped)
            if page_match:
                flush()
                current_page = int(page_match.group(1))
                continue

            heading_match = _HEADING_RE.match(stripped)
            if heading_match:
                flush()
                heading_text = heading_match.group(2).strip()
                current_section = heading_text
                blocks.append(
                    _Block(
                        type="heading",
                        text=heading_text,
                        page=current_page,
                        section=heading_text,
                        level=len(heading_match.group(1)),
                    )
                )
                continue

            if _TABLE_ROW_RE.match(stripped):
                if buffer_type != "table":
                    flush()
                    buffer_type = "table"
                buffer.append(stripped)
                continue

            if _LIST_ITEM_RE.match(stripped):
                if buffer_type != "list":
                    flush()
                    buffer_type = "list"
                buffer.append(stripped)
                continue

            if _QUOTE_RE.match(stripped):
                if buffer_type != "quote":
                    flush()
                    buffer_type = "quote"
                buffer.append(stripped)
                continue

            # Plain paragraph line.
            if buffer_type not in (None, "paragraph"):
                flush()
            buffer_type = "paragraph"
            buffer.append(stripped)

        flush()
        return blocks

    @staticmethod
    def _split_oversized_blocks(blocks: list[_Block]) -> list[_Block]:
        """
        A single paragraph/list/quote block can itself be longer than
        STRUCTURE_CHUNK_MAX_CHARS (e.g. one dense unbroken paragraph on a
        page) — _group_blocks only ever breaks *between* blocks, so
        without this step that block would sail through as one oversized
        chunk. Split any such block into sliding-window sub-blocks
        (overlap = PAGE_CHUNK_OVERLAP_CHARS), preserving its page/section.

        Tables and headings are left untouched: tables must never be
        split (hard rule elsewhere), and headings are always short.
        """
        result: list[_Block] = []

        for block in blocks:
            if block.type in ("table", "heading") or len(block.text) <= STRUCTURE_CHUNK_MAX_CHARS:
                result.append(block)
                continue

            step = max(1, STRUCTURE_CHUNK_MAX_CHARS - PAGE_CHUNK_OVERLAP_CHARS)
            text = block.text
            start = 0
            while start < len(text):
                end = min(len(text), start + STRUCTURE_CHUNK_MAX_CHARS)

                # Prefer breaking at a sentence/line boundary near the end.
                if end < len(text):
                    window = text[start:end]
                    last_period = window.rfind(". ")
                    last_newline = window.rfind("\n")
                    break_at = max(last_period, last_newline)
                    if break_at > len(window) - PAGE_CHUNK_OVERLAP_CHARS:
                        end = start + break_at + 1

                piece = text[start:end].strip()
                if piece:
                    result.append(
                        _Block(
                            type=block.type,
                            text=piece,
                            page=block.page,
                            section=block.section,
                        )
                    )

                if end >= len(text):
                    break
                start = max(start + 1, end - PAGE_CHUNK_OVERLAP_CHARS)

        return result

    @staticmethod
    def _group_blocks(blocks: list[_Block]) -> list[list[_Block]]:
        """
        Group blocks into page-wise chunks, applying hard breaks in this
        priority order:

          1. A table block is always its own chunk (never split, never
             merged with surrounding text) — this is what keeps a
             comparison table's rows and columns intact and retrievable
             as one coherent unit.
          2. A page boundary always starts a new chunk. This is what makes
             chunking genuinely page-wise: a chunk can never silently span
             two source pages, which keeps `page` metadata exact and keeps
             page-level citations honest. (Documents with no page markers
             at all — e.g. plain text/markdown — have `block.page is None`
             throughout, so this rule is a no-op for them and grouping
             falls back to the char-budget/heading rules below.)
          3. A heading always starts a new chunk, so a chunk reads as
             "heading + the content under it" rather than an arbitrary
             slice of running text.

        Within a page, if content still exceeds STRUCTURE_CHUNK_MAX_CHARS,
        the page is split further using the same char-budget rule — this
        is what keeps a very dense page from becoming one giant chunk.
        """
        groups: list[list[_Block]] = []
        current: list[_Block] = []
        current_len = 0
        current_page: int | None = None

        def flush_group() -> None:
            nonlocal current, current_len, current_page
            if current:
                groups.append(current)
            current, current_len, current_page = [], 0, None

        for block in blocks:
            block_len = len(block.text)

            # Hard break #1: table is always its own chunk.
            if block.type == "table":
                flush_group()
                groups.append([block])
                continue

            # Hard break #2: a new page always starts a new chunk.
            if (
                current
                and block.page is not None
                and current_page is not None
                and block.page != current_page
            ):
                flush_group()

            # Hard break #3: a heading always starts a new chunk.
            if block.type == "heading" and current:
                flush_group()

            # Soft break: char budget exceeded — splits an overlong page.
            if current and current_len + block_len > STRUCTURE_CHUNK_MAX_CHARS:
                flush_group()

            current.append(block)
            current_len += block_len
            if block.page is not None:
                current_page = block.page

        flush_group()
        return groups

    @staticmethod
    def _build_prose_nodes(groups: list[list[_Block]]) -> list[BaseNode]:
        rendered = ["\n\n".join(b.render() for b in group) for group in groups]

        nodes: list[BaseNode] = []
        for i, group in enumerate(groups):
            curr_text = rendered[i]

            # Sliding-window overlap: a trimmed slice of the previous/next
            # chunk rather than the whole neighboring chunk, so the window
            # genuinely reflects "a bit of context either side" instead of
            # ballooning to 3x the chunk size on every hop.
            prev_overlap = rendered[i - 1][-PAGE_CHUNK_OVERLAP_CHARS:] if i > 0 else ""
            next_overlap = rendered[i + 1][:PAGE_CHUNK_OVERLAP_CHARS] if i < len(rendered) - 1 else ""

            window_text = "\n\n".join(p for p in (prev_overlap, curr_text, next_overlap) if p)

            first_block = group[0]
            pages_in_group = sorted({b.page for b in group if b.page is not None})

            node = TextNode(
                text=curr_text,
                metadata={
                    "window": window_text,
                    "original_text": curr_text,
                    "page": first_block.page,
                    "page_range": pages_in_group or None,
                    "section": first_block.section,
                },
            )
            nodes.append(node)

        return nodes

    # ── Private helpers — code ───────────────────────────────────────────────

    def _chunk_code(self, text: str, language: str) -> list[BaseNode]:
        """
        Apply CodeSplitter to code files, then augment each chunk with a
        "window" field containing the preceding + current + following chunk text.
        """
        try:
            splitter = CodeSplitter(
                language=language,
                chunk_lines=settings.CODE_CHUNK_LINES,
                chunk_lines_overlap=settings.CODE_CHUNK_OVERLAP,
                max_chars=settings.CODE_CHUNK_MAX_CHARS,
            )
            from llama_index.core.schema import Document as LlamaDocument

            llama_doc = LlamaDocument(text=text)
            nodes: list[BaseNode] = splitter.get_nodes_from_documents([llama_doc])
            logger.info(
                "Code split complete",
                extra={"language": language, "node_count": len(nodes)},
            )
        except Exception as exc:
            logger.warning(
                "CodeSplitter failed; falling back to prose chunking",
                extra={"language": language, "error": str(exc)},
            )
            return self._chunk_prose(text)

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
