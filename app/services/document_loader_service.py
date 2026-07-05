"""
Document loader service — structured text extraction from PDF and DOCX.

PDF extraction pipeline (in order — first tier to produce meaningful text wins):

  1. Native text-layer extraction
       a. PyMuPDF4LLM  — best structure preservation (headings, tables, lists)
       b. pdfplumber   — strong table extraction via extract_tables()
       c. pypdf        — last-resort plain text extraction
     These all read the PDF's embedded font/text objects. If the PDF has
     NO embedded text at all — e.g. a scanned document, or a design-tool
     export (Figma/Keynote/Illustrator) where every glyph was "outlined"
     to vector curves before export — all three return empty text, even
     though the page renders perfectly. `pdffonts <file>` will show zero
     font objects in that case; that's the signal this tier can't help.

  2. PaddleOCR — PP-StructureV3 (primary fallback, fully local/offline)
       Each page is rasterized to an image (pdf2image/poppler) and run
       through PaddleOCR's PP-StructureV3 pipeline, which performs layout
       detection, OCR, and table-structure recognition together and
       returns ready-made Markdown per page — headings, paragraphs, and
       tables already reconstructed in correct reading order, including
       multi-column pages.

       This replaced an earlier Groq vision-LLM tier. Functionally it does
       the same job (page image in, structured Markdown out) but runs
       entirely on this machine — no API key, no per-page network call,
       no external provider in the loop. PP-StructureV3 was chosen over
       plain tesseract for exactly the same reason the old vision-LLM tier
       was chosen over tesseract: tesseract has no layout model, so it
       reliably scrambles multi-column pages and mishandles text sitting
       on colored/highlighted backgrounds. PP-StructureV3 reads the page
       the way a layout-aware model does, keeping columns, tables, and
       heading hierarchy intact.

       One-time cost: the first time PP-StructureV3 runs in a given
       environment it downloads its model weights (a few hundred MB) from
       Paddle's model hub. That download needs real internet access once;
       every run after that is fully offline. If that download can't
       happen (air-gapped box, no internet yet), set PADDLEOCR_ENABLED=False
       and this tier is skipped in favor of tesseract below.

  3. Tesseract OCR (last-resort fallback)
       Used if PADDLEOCR_ENABLED=False, PaddleOCR isn't installed, or the
       PaddleOCR pass raises for every page. Cheap, but lower fidelity: no
       real layout awareness, known to scramble multi-column reading order
       and drop text on colored backgrounds.

DOCX pipeline (unchanged): python-docx block-order walk — paragraphs
(with heading levels) and tables interleaved in document order, tables
rendered as Markdown.

New dependencies for tier 2 (add to requirements.txt):
    paddleocr        (pulls in the PP-StructureV3 pipeline)
    paddlepaddle     (CPU wheel is fine to start; swap for the GPU wheel
                       if you have CUDA available — see PaddlePaddle's own
                       install docs for the right index URL)
    pdf2image        (already used by tier 3; needs poppler-utils installed)
    numpy

New settings expected on `app.core.config.settings` (all optional — falls
back to sane defaults / skips the tier if unset):
    PADDLEOCR_ENABLED   — default True
    PADDLEOCR_LANG      — default "en"
    PADDLEOCR_DPI       — default 200
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import IngestionFailedError
from app.core.logger import get_logger

logger = get_logger(__name__)

# Shared prose types — all use structure-aware chunking downstream.
PROSE_FILE_TYPES = frozenset({"pdf", "docx", "text", "markdown", "txt", "md"})

# Native text threshold before we decide a PDF needs a fallback tier.
MIN_NATIVE_PDF_TEXT_CHARS = settings.PDF_NATIVE_TEXT_MIN_CHARS
OCR_DPI = settings.OCR_DPI

# PaddleOCR (PP-StructureV3) tier settings — read defensively so this file
# doesn't hard-fail if config.py hasn't been updated with these yet.
PADDLEOCR_ENABLED = getattr(settings, "PADDLEOCR_ENABLED", True)
PADDLEOCR_LANG = getattr(settings, "PADDLEOCR_LANG", "en")
PADDLEOCR_DPI = getattr(settings, "PADDLEOCR_DPI", 200)

# Module-level singleton — PP-StructureV3 model loading is expensive
# (multiple sub-models: layout detection, text detection/recognition,
# table structure recognition). Load once per process, reuse for every
# page of every document handled by this worker.
_PADDLE_PIPELINE = None


class DocumentLoaderService:
    """Extract plain/Markdown text from supported document formats."""

    def load(self, path: str | Path, file_type: str) -> str:
        path = Path(path)
        if not path.exists():
            raise IngestionFailedError(str(path), f"File not found: {path}")

        file_type = (file_type or "").lower().strip()

        if file_type == "pdf":
            return self._load_pdf(path)

        if file_type == "docx":
            return self._load_docx(path)

        # For text-like files, just read content directly.
        return path.read_text(encoding="utf-8", errors="replace")

    # ── PDF ────────────────────────────────────────────────────────────────────

    def _load_pdf(self, path: Path) -> str:
        """
        Try native extraction first. If the result is empty or too small,
        fall back to local PaddleOCR (PP-StructureV3), then to tesseract.
        """
        native_loaders = (
            ("pymupdf4llm", self._load_pdf_pymupdf4llm),
            ("pdfplumber", self._load_pdf_pdfplumber),
            ("pypdf", self._load_pdf_pypdf),
        )

        errors: list[str] = []

        for name, loader in native_loaders:
            try:
                text = loader(path)
                if self._is_meaningful_text(text):
                    logger.info(
                        "PDF loaded",
                        extra={"loader": name, "path": str(path), "chars": len(text)},
                    )
                    return text

                errors.append(
                    f"{name}: {'insufficient text (' + str(len(text)) + ' chars)' if text.strip() else 'empty output'}"
                )
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                logger.warning(
                    "PDF loader failed",
                    extra={"loader": name, "path": str(path), "error": str(exc)},
                )

        # Native extraction failed or produced low-quality output — expected
        # for scanned PDFs and for design-tool exports with no embedded text
        # layer (outlined/vectorized text). Try local PaddleOCR next; it's
        # the primary fallback, not tesseract.
        if PADDLEOCR_ENABLED:
            try:
                text = self._load_pdf_paddleocr(path)
                if self._is_meaningful_text(text):
                    logger.info(
                        "PDF loaded via PaddleOCR (PP-StructureV3)",
                        extra={"loader": "paddleocr", "path": str(path), "chars": len(text)},
                    )
                    return text
                errors.append(f"paddleocr: insufficient text ({len(text)} chars)")
            except Exception as exc:
                errors.append(f"paddleocr: {exc}")
                logger.warning(
                    "PaddleOCR extraction failed; falling back to tesseract",
                    extra={"path": str(path), "error": str(exc)},
                )
        else:
            logger.warning(
                "PADDLEOCR_ENABLED=False; skipping PaddleOCR tier and falling back "
                "directly to tesseract (lower fidelity)",
                extra={"path": str(path)},
            )

        # Last resort — cheap, but no real layout awareness. Known to scramble
        # multi-column pages and drop text on colored/highlighted boxes.
        try:
            text = self._load_pdf_pytesseract(path)
            if self._is_meaningful_text(text):
                logger.info(
                    "PDF loaded via tesseract OCR (last-resort fallback)",
                    extra={"loader": "tesseract", "path": str(path), "chars": len(text)},
                )
                return text
            errors.append(f"tesseract: insufficient text ({len(text)} chars)")
        except Exception as exc:
            errors.append(f"tesseract: {exc}")
            logger.warning(
                "Tesseract OCR failed",
                extra={"path": str(path), "error": str(exc)},
            )

        raise IngestionFailedError(
            str(path),
            "Could not extract text from PDF. Tried: " + "; ".join(errors),
        )

    @staticmethod
    def _load_pdf_pymupdf4llm(path: Path) -> str:
        import pymupdf4llm

        text = pymupdf4llm.to_markdown(str(path))
        return _clean_pdf_text(text)

    @staticmethod
    def _load_pdf_pdfplumber(path: Path) -> str:
        import pdfplumber

        parts: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                parts.append(f"## Page {page_num}")

                page_text = (page.extract_text() or "").strip()
                if page_text:
                    parts.append(page_text)

                for table in page.extract_tables() or []:
                    md_table = table_to_markdown(table)
                    if md_table:
                        parts.append(md_table)

        text = "\n\n".join(parts)
        return _clean_pdf_text(text)

    @staticmethod
    def _load_pdf_pypdf(path: Path) -> str:
        import pypdf

        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)
        return _clean_pdf_text(text)

    # ── PaddleOCR / PP-StructureV3 (primary fallback, fully local) ─────────────

    def _load_pdf_paddleocr(self, path: Path) -> str:
        """
        Transcribe every page locally via PaddleOCR's PP-StructureV3
        pipeline — layout detection + OCR + table recognition + reading
        order recovery, all running on this machine. No external API call.

        Each page is rasterized (pdf2image/poppler, same rendering path
        tesseract already uses) and handed to PP-StructureV3, which
        returns ready-made Markdown per page. Joined with '## Page N'
        markers so downstream chunking recovers page numbers exactly like
        the native-extraction tiers do.
        """
        import numpy as np
        from pdf2image import convert_from_path

        pipeline = self._get_paddle_pipeline()
        pages = convert_from_path(str(path), dpi=PADDLEOCR_DPI)

        parts: list[str] = []
        for i, page_image in enumerate(pages, start=1):
            try:
                page_array = np.array(page_image.convert("RGB"))
                page_md_parts: list[str] = []
                for res in pipeline.predict(page_array):
                    md = getattr(res, "markdown", None)
                    md_text = (md or {}).get("markdown_texts", "") if md else ""
                    if md_text:
                        page_md_parts.append(md_text)
                page_text = "\n\n".join(page_md_parts).strip()
            except Exception as exc:
                logger.warning(
                    "PaddleOCR failed for a single page; continuing with remaining pages",
                    extra={"page": i, "path": str(path), "error": str(exc)},
                )
                page_text = ""

            parts.append(f"## Page {i}\n\n{page_text}".strip())

        return _clean_pdf_text("\n\n".join(parts))

    @staticmethod
    def _get_paddle_pipeline():
        """
        Lazily construct and cache the PP-StructureV3 pipeline as a
        module-level singleton. Model loading is expensive (layout
        detection + text detection/recognition + table-structure models),
        so this happens once per worker process and is reused for every
        page of every document after that.
        """
        global _PADDLE_PIPELINE
        if _PADDLE_PIPELINE is None:
            from paddleocr import PPStructureV3

            logger.info("Loading PP-StructureV3 pipeline", extra={"lang": PADDLEOCR_LANG})
            _PADDLE_PIPELINE = PPStructureV3(
                lang=PADDLEOCR_LANG,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            logger.info("PP-StructureV3 pipeline loaded")
        return _PADDLE_PIPELINE

    # ── Tesseract OCR (last-resort fallback) ────────────────────────────────────

    def _load_pdf_pytesseract(self, path: Path) -> str:
        import pytesseract
        from pdf2image import convert_from_path

        missing_tools = [
            tool
            for tool in ("pdfinfo", "pdftoppm", "tesseract")
            if shutil.which(tool) is None
        ]
        if missing_tools:
            raise RuntimeError(
                "OCR dependencies missing from PATH: "
                + ", ".join(missing_tools)
                + ". Install Poppler and Tesseract "
                "(Ubuntu/WSL: sudo apt install poppler-utils tesseract-ocr)."
            )

        pages = convert_from_path(str(path), dpi=OCR_DPI)
        parts: list[str] = []

        for i, page in enumerate(pages, start=1):
            parts.append(f"## Page {i}")

            # --psm 4 assumes a single column of text of variable sizes,
            # which handles multi-block layouts somewhat better than the
            # default "fully automatic" mode. Still no real layout model —
            # this is a last resort, not a fix.
            page_text = pytesseract.image_to_string(page, config="--psm 4")
            page_text = _clean_pdf_text(page_text)

            if page_text:
                parts.append(page_text)

        return _clean_pdf_text("\n\n".join(parts))

    # ── DOCX ───────────────────────────────────────────────────────────────────

    def _load_docx(self, path: Path) -> str:
        try:
            text = self._load_docx_python_docx(path)
            if text.strip():
                logger.info(
                    "DOCX loaded",
                    extra={"loader": "python-docx", "path": str(path), "chars": len(text)},
                )
                return text
        except Exception as exc:
            logger.warning(
                "python-docx loader failed",
                extra={"path": str(path), "error": str(exc)},
            )
            raise IngestionFailedError(
                str(path), f"Could not extract text from DOCX: {exc}"
            ) from exc

        raise IngestionFailedError(str(path), "DOCX produced no extractable content")

    @staticmethod
    def _load_docx_python_docx(path: Path) -> str:
        from docx import Document
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        doc = Document(path)
        parts: list[str] = []

        for child in doc.element.body:
            if child.tag == qn("w:p"):
                para = Paragraph(child, doc)
                text = para.text.strip()
                if not text:
                    continue
                parts.append(format_paragraph(para))

            elif child.tag == qn("w:tbl"):
                table = Table(child, doc)
                md = docx_table_to_markdown(table)
                if md:
                    parts.append(md)

        return "\n\n".join(parts)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_meaningful_text(text: str) -> bool:
        """
        Decide whether extracted text is good enough to skip the next
        fallback tier.

        This filters out results that technically return some text but are
        mostly empty, garbled, or just page numbers.
        """
        if not text:
            return False

        cleaned = text.strip()
        if len(cleaned) < MIN_NATIVE_PDF_TEXT_CHARS:
            return False

        alpha_count = sum(ch.isalpha() for ch in cleaned)
        digit_count = sum(ch.isdigit() for ch in cleaned)

        # Require a reasonable amount of real text.
        if alpha_count < max(50, MIN_NATIVE_PDF_TEXT_CHARS // 4):
            return False

        # Avoid accepting output that is mostly numbers / noise.
        if digit_count > alpha_count * 4 and len(cleaned) < 1000:
            return False

        return True


def table_to_markdown(table: list[list | tuple] | None) -> str:
    """Convert a row/column grid to a GitHub-flavoured Markdown table."""
    if not table:
        return ""

    rows: list[list[str]] = []
    for row in table:
        if row is None:
            continue
        rows.append([_escape_md_cell(str(cell or "").replace("\n", " ").strip()) for cell in row])

    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    normalized: list[list[str]] = []
    for row in rows:
        padded = row + [""] * (col_count - len(row))
        normalized.append(padded[:col_count])

    header = normalized[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def docx_table_to_markdown(table) -> str:
    """Convert a python-docx Table to Markdown."""
    rows: list[list[str]] = []
    for row in table.rows:
        rows.append([_escape_md_cell(cell.text.replace("\n", " ").strip()) for cell in row.cells])
    return table_to_markdown(rows)


def format_paragraph(para) -> str:
    """Render a DOCX paragraph, preserving heading levels when detectable."""
    text = para.text.strip()
    style_name = (para.style.name or "").lower() if para.style else ""

    if not text:
        return ""

    if "heading 1" in style_name or style_name == "title":
        return f"# {text}"
    if "heading 2" in style_name:
        return f"## {text}"
    if "heading 3" in style_name:
        return f"### {text}"
    if "heading 4" in style_name:
        return f"#### {text}"
    if "heading 5" in style_name:
        return f"##### {text}"
    if "heading 6" in style_name:
        return f"###### {text}"

    if "list" in style_name or "bullet" in style_name:
        return f"- {text}"

    return text


def _escape_md_cell(text: str) -> str:
    """Escape Markdown table-breaking characters."""
    return text.replace("|", "\\|").strip()


def _clean_pdf_text(text: str) -> str:
    """
    Clean common PDF/OCR extraction artifacts.

    Handles:
      - Null bytes
      - Control characters
      - Excessive whitespace
      - Excessive blank lines
    """
    if not text:
        return ""

    # Remove null bytes and normalize common encoding junk.
    text = text.replace("\x00", "")
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    # Remove control characters except tab/newline/carriage return.
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Trim trailing spaces on each line and collapse repeated spaces.
    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        line = " ".join(line.split())
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()