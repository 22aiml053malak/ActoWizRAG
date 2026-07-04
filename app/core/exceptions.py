"""
Custom exception classes and FastAPI exception handlers.

All expected failure modes map to a specific HTTP status code so that callers
never encounter a bare 500 for business-logic errors.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import traceback

from app.core.logger import get_logger

logger = get_logger(__name__)


# ── Domain exceptions ──────────────────────────────────────────────────────────

class ActowizBaseError(Exception):
    """Base class for all application-specific exceptions."""
    http_status: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or message


class DocumentNotFoundError(ActowizBaseError):
    http_status = 404
    error_code = "DOCUMENT_NOT_FOUND"

    def __init__(self, document_id: str) -> None:
        super().__init__(
            message=f"Document '{document_id}' not found.",
            detail=f"No document with id={document_id} exists (or it has been deleted).",
        )


class UnsupportedFileTypeError(ActowizBaseError):
    http_status = 422
    error_code = "UNSUPPORTED_FILE_TYPE"

    def __init__(self, filename: str, allowed: list[str]) -> None:
        super().__init__(
            message=f"File type not supported for '{filename}'.",
            detail=f"Allowed extensions: {allowed}",
        )


class IngestionFailedError(ActowizBaseError):
    http_status = 500
    error_code = "INGESTION_FAILED"

    def __init__(self, document_id: str, reason: str) -> None:
        super().__init__(
            message=f"Ingestion failed for document '{document_id}'.",
            detail=reason,
        )


class DocumentAlreadyDeletingError(ActowizBaseError):
    http_status = 409
    error_code = "DOCUMENT_ALREADY_DELETING"

    def __init__(self, document_id: str) -> None:
        super().__init__(
            message=f"Document '{document_id}' is already being deleted.",
        )


class VectorStoreError(ActowizBaseError):
    http_status = 500
    error_code = "VECTOR_STORE_ERROR"


class LLMProviderError(ActowizBaseError):
    http_status = 502
    error_code = "LLM_PROVIDER_ERROR"

    def __init__(self, provider: str, reason: str) -> None:
        super().__init__(
            message=f"LLM provider '{provider}' returned an error.",
            detail=reason,
        )


class LLMProviderNotConfiguredError(ActowizBaseError):
    http_status = 501
    error_code = "LLM_PROVIDER_NOT_CONFIGURED"

    def __init__(self) -> None:
        super().__init__(
            message="No LLM provider is configured.",
            detail=(
                "Set LLM_PROVIDER env var to 'groq' or 'openai_compatible' "
                "and supply a valid LLM_API_KEY to enable answer generation."
            ),
        )


# ── FastAPI exception handlers ─────────────────────────────────────────────────

def _error_response(
    status_code: int,
    error_code: str,
    message: str,
    detail: str | None = None,
) -> JSONResponse:
    body = {"error": error_code, "message": message}
    if detail:
        body["detail"] = detail
    return JSONResponse(status_code=status_code, content=body)


def register_exception_handlers(app: FastAPI) -> None:
    """Register all custom exception handlers on the FastAPI application."""

    @app.exception_handler(ActowizBaseError)
    async def actowiz_error_handler(
        request: Request, exc: ActowizBaseError
    ) -> JSONResponse:
        logger.warning(
            "Domain exception",
            extra={
                "error_code": exc.error_code,
                "message": exc.message,
                "path": str(request.url),
            },
        )
        return _error_response(
            status_code=exc.http_status,
            error_code=exc.error_code,
            message=exc.message,
            detail=exc.detail,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.info(
            "Request validation error",
            extra={"errors": exc.errors(), "path": str(request.url)},
        )
        return _error_response(
            status_code=422,
            error_code="VALIDATION_ERROR",
            message="Request validation failed.",
            detail=str(exc.errors()),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            "Unhandled exception",
            extra={
                "path": str(request.url),
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        return _error_response(
            status_code=500,
            error_code="INTERNAL_ERROR",
            message="An unexpected internal error occurred.",
        )
