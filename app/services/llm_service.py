"""
LLM Service — provider-agnostic AI Gateway.

This is the SINGLE place in the codebase that talks to external LLM APIs.
rag_service.py calls llm_service; never a provider SDK directly.

Adding a new provider:
  1. Implement the LLMProvider Protocol.
  2. Add a branch in `get_llm_provider()`.
  3. Update LLM_PROVIDER enum in core/config.py.

Answer-generation quality notes (why the code below looks the way it does):
  - Retrieved chunks carry `page`/`section` metadata recovered by
    chunking_service.py from `## Page N` markers and headings. We surface
    both in the context header so the model can actually cite a real page
    instead of just a filename — and so a human can go verify it.
  - This pipeline's text ultimately comes from OCR (PaddleOCR/tesseract) for
    scanned or design-tool-exported PDFs, not just clean native extraction.
    OCR output reliably contains small recognition errors (e.g. "Al" for
    "AI", merged words, stray glyphs from icons/checkmarks). The prompt
    tells the model this explicitly, so it reads through minor noise
    instead of either propagating typos verbatim or refusing to engage
    with a slightly garbled paragraph.
  - A low-relevance chunk (e.g. a bad-OCR page that barely matched the
    query) is worse than no chunk — it invites the model to stretch for an
    answer. MIN_RELEVANCE_SCORE filters those out before the prompt is
    even built.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx

from app.core.config import settings
from app.core.exceptions import LLMProviderError, LLMProviderNotConfiguredError
from app.core.logger import get_logger
from app.models.response import ChunkResult

logger = get_logger(__name__)

# Chunks scoring below this cosine similarity are dropped before the prompt.
# Cosine similarity from PGVectorStore is in [0, 1]; 0.2 is deliberately
# permissive so borderline matches still reach the LLM.
MIN_RELEVANCE_SCORE = getattr(settings, "MIN_RELEVANCE_SCORE", 0.20)

# Floor for RAG answer synthesis specifically. Summarising several reranked
# chunks *and* citing sources needs more headroom than a bare completion —
# if config.py's general LLM_MAX_TOKENS is set lower than this for some
# other use of the same settings, this call still gets enough room to finish
# instead of getting cut off mid-answer.
MIN_ANSWER_MAX_TOKENS = 1536


# ── Protocol ───────────────────────────────────────────────────────────────────

@runtime_checkable
class LLMProvider(Protocol):
    """Provider-agnostic interface for LLM text completion."""

    def complete(self, prompt: str, **kwargs: object) -> str:
        """Return a completion string for the given prompt."""
        ...


# ── Providers ──────────────────────────────────────────────────────────────────

class GroqProvider:
    """
    Groq Cloud via their OpenAI-compatible REST API.

    Env vars used: LLM_API_KEY, LLM_MODEL_NAME, LLM_MAX_TOKENS, LLM_TEMPERATURE.
    """

    def __init__(self) -> None:
        if not settings.LLM_API_KEY:
            raise LLMProviderError("groq", "LLM_API_KEY is not set")
        self._client = httpx.Client(
            base_url=settings.LLM_API_BASE_URL.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
            timeout=60.0,
        )

    def complete(self, prompt: str, **kwargs: object) -> str:
        payload = {
            "model": settings.LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
            "temperature": kwargs.get("temperature", settings.LLM_TEMPERATURE),
        }
        try:
            resp = self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                "groq", f"HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise LLMProviderError("groq", str(exc)) from exc


class OpenAICompatibleProvider:
    """
    Generic OpenAI-compatible provider (OpenAI, Azure, Together, etc.).

    Env vars used: LLM_API_KEY, LLM_API_BASE_URL, LLM_MODEL_NAME,
                   LLM_MAX_TOKENS, LLM_TEMPERATURE.
    """

    def __init__(self) -> None:
        if not settings.LLM_API_KEY:
            raise LLMProviderError("openai_compatible", "LLM_API_KEY is not set")
        self._client = httpx.Client(
            base_url=settings.LLM_API_BASE_URL.rstrip("/"),
            headers={
                "Authorization": f"Bearer {settings.LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    def complete(self, prompt: str, **kwargs: object) -> str:
        payload = {
            "model": settings.LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
            "temperature": kwargs.get("temperature", settings.LLM_TEMPERATURE),
        }
        try:
            resp = self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                "openai_compatible", f"HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise LLMProviderError("openai_compatible", str(exc)) from exc


# ── Factory ────────────────────────────────────────────────────────────────────

def get_llm_provider() -> LLMProvider:
    """
    Return the active LLM provider based on the LLM_PROVIDER env var.

    Raises LLMProviderNotConfiguredError if LLM_PROVIDER='none'.
    """
    provider = settings.LLM_PROVIDER
    if provider == "none":
        raise LLMProviderNotConfiguredError()
    if provider == "groq":
        return GroqProvider()
    if provider == "openai_compatible":
        return OpenAICompatibleProvider()
    raise LLMProviderError(provider, f"Unknown LLM_PROVIDER value: {provider!r}")


# ── LLM Service ────────────────────────────────────────────────────────────────

class LLMService:
    """
    Coordinates prompt construction and provider dispatch for RAG answer generation.
    """

    def generate_answer(
        self,
        query: str,
        chunks: list[ChunkResult],
    ) -> tuple[str, list[str]]:
        """
        Synthesise an answer given a query and the retrieved + windowed chunks.

        Chunks scoring below MIN_RELEVANCE_SCORE are dropped before the
        prompt is built. If nothing survives that filter, we return an
        honest "not found" answer without spending a provider call.

        Returns:
            (answer_text, list_of_source_filenames)
        """
        relevant = [c for c in chunks if (c.score or 0.0) >= MIN_RELEVANCE_SCORE]

        if not relevant:
            logger.info(
                "No chunks met the relevance threshold; skipping LLM call",
                extra={
                    "query": query[:80],
                    "candidates": len(chunks),
                    "min_relevance_score": MIN_RELEVANCE_SCORE,
                },
            )
            return (
                "No sufficiently relevant content was found in the knowledge base "
                "to answer this question.",
                [],
            )

        provider = get_llm_provider()

        # Build a structured RAG prompt. Each block is labelled with as much
        # real location info as we have — filename, page, section — so the
        # model can cite something a human can actually go check.
        context_blocks: list[str] = []
        sources: list[str] = []
        for i, chunk in enumerate(relevant, start=1):
            fname = chunk.filename or "unknown"
            if fname not in sources:
                sources.append(fname)

            meta = chunk.metadata or {}
            page = meta.get("page")
            section = meta.get("section")

            location_parts = [fname]
            if page:
                location_parts.append(f"p.{page}")
            if section:
                location_parts.append(section)
            location = " — ".join(location_parts)

            context_blocks.append(f"[{i}] (source: {location})\n{chunk.content}")

        context = "\n\n---\n\n".join(context_blocks)
        prompt = (
            "You are a helpful technical assistant. "
            "Answer the question below using ONLY the provided context. "
            "If the context does not contain enough information to answer, "
            "say so explicitly. Cite the source numbers [1], [2], etc. where relevant.\n\n"
            "The context was extracted via OCR from PDF pages and may contain minor "
            "recognition errors — e.g. 'Al' for 'AI', merged or duplicated words, or "
            "stray characters from icons and checkmarks. Read through such errors "
            "using your best judgement; do not treat them as meaningful content, and "
            "do not invent information that isn't actually present in the context.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER:"
        )

        logger.info(
            "Sending prompt to LLM",
            extra={
                "provider": settings.LLM_PROVIDER,
                "model": settings.LLM_MODEL_NAME,
                "context_chunks": len(relevant),
                "dropped_low_relevance": len(chunks) - len(relevant),
            },
        )

        max_tokens = max(settings.LLM_MAX_TOKENS, MIN_ANSWER_MAX_TOKENS)
        answer = provider.complete(prompt, max_tokens=max_tokens)
        return answer.strip(), sources
