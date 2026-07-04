"""
LLM Service — provider-agnostic AI Gateway.

This is the SINGLE place in the codebase that talks to external LLM APIs.
rag_service.py calls llm_service; never a provider SDK directly.

Adding a new provider:
  1. Implement the LLMProvider Protocol.
  2. Add a branch in `get_llm_provider()`.
  3. Update LLM_PROVIDER enum in core/config.py.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

import httpx

from app.core.config import settings
from app.core.exceptions import LLMProviderError, LLMProviderNotConfiguredError
from app.core.logger import get_logger
from app.models.response import ChunkResult

logger = get_logger(__name__)


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
            base_url="https://api.groq.com/openai/v1",
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

        Returns:
            (answer_text, list_of_source_filenames)
        """
        provider = get_llm_provider()

        # Build a structured RAG prompt.
        context_blocks = []
        sources: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            fname = chunk.filename or "unknown"
            if fname not in sources:
                sources.append(fname)
            context_blocks.append(
                f"[{i}] (source: {fname})\n{chunk.content}"
            )

        context = "\n\n---\n\n".join(context_blocks)
        prompt = (
            "You are a helpful technical assistant. "
            "Answer the question below using ONLY the provided context. "
            "If the context does not contain enough information to answer, "
            "say so explicitly. Cite the source numbers [1], [2], etc. where relevant.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER:"
        )

        logger.info(
            "Sending prompt to LLM",
            extra={
                "provider": settings.LLM_PROVIDER,
                "model": settings.LLM_MODEL_NAME,
                "context_chunks": len(chunks),
            },
        )

        answer = provider.complete(prompt)
        return answer.strip(), sources
