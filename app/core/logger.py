"""
Structured logging module.

All modules should call `get_logger(__name__)` to obtain their logger.
Log records are emitted as JSON lines in production for easy ingestion by
log aggregators (Datadog, Loki, Cloud Logging, etc.).
"""

import logging
import sys
import json
from datetime import datetime, timezone
from typing import Any


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    RESERVED_ATTRS = frozenset(
        {
            "args", "asctime", "created", "exc_info", "exc_text",
            "filename", "funcName", "levelname", "levelno", "lineno",
            "message", "module", "msecs", "msg", "name", "pathname",
            "process", "processName", "relativeCreated", "stack_info",
            "thread", "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        extra: dict[str, Any] = {
            k: v
            for k, v in record.__dict__.items()
            if k not in self.RESERVED_ATTRS and not k.startswith("_")
        }

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if extra:
            payload["context"] = extra

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Usage::

        from app.core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("document ingested", extra={"document_id": doc_id, "chunk_count": n})
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Use JSON formatter in all environments for consistency;
        # developers can set LOG_LEVEL=DEBUG for verbose output.
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def configure_root_logger(log_level: str = "INFO") -> None:
    """Call once at application startup (main.py) to set the root log level."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.root.setLevel(numeric_level)
    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
