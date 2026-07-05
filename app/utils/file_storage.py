"""
File storage utility.

Saves uploaded files to a content-addressed path under STORAGE_DIR:
  <STORAGE_DIR>/<document_id>/<original_filename>

This avoids collisions (two files called "report.pdf" for different docs)
while keeping the original filename readable for debugging.
"""

import os
import shutil
from pathlib import Path

import aiofiles
from fastapi import UploadFile

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# Allowed extensions → normalised file_type label
ALLOWED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "text",
    ".md": "markdown",
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".java": "code",
    ".go": "code",
    ".rs": "code",
    ".cpp": "code",
    ".c": "code",
    ".cs": "code",
    ".rb": "code",
    ".php": "code",
    ".sh": "code",
    ".yaml": "text",
    ".yml": "text",
    ".json": "text",
    ".toml": "text",
}

# Map extension → tree-sitter language name (for CodeSplitter)
LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "bash",
}


def get_file_type(filename: str) -> str:
    """Return normalised file_type or raise UnsupportedFileTypeError."""
    from app.core.exceptions import UnsupportedFileTypeError

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            filename=filename,
            allowed=list(ALLOWED_EXTENSIONS.keys()),
        )
    return ALLOWED_EXTENSIONS[ext]


def get_language(filename: str) -> str | None:
    """Return the tree-sitter language name for a code file, or None."""
    ext = Path(filename).suffix.lower()
    return LANGUAGE_MAP.get(ext)


async def save_upload(file: UploadFile, document_id: str) -> str:
    """
    Persist an uploaded file to disk.

    Returns the absolute storage path.
    """
    dest_dir = Path(settings.STORAGE_DIR) / document_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Sanitise filename — strip directory traversal attempts.
    safe_name = Path(file.filename or "upload").name
    dest_path = dest_dir / safe_name

    async with aiofiles.open(dest_path, "wb") as out_file:
        # Stream in 64 KB chunks to avoid loading the entire file into RAM.
        while chunk := await file.read(65_536):
            await out_file.write(chunk)

    logger.info(
        "File saved",
        extra={"document_id": document_id, "path": str(dest_path), "file_name": safe_name},
    )
    return str(dest_path)


def read_file_bytes(storage_path: str) -> bytes:
    """Read and return the raw bytes of a stored file."""
    with open(storage_path, "rb") as f:
        return f.read()


def delete_file(storage_path: str) -> None:
    """Remove a file (and its parent directory) from disk.  Silent if missing."""
    path = Path(storage_path)
    try:
        if path.exists():
            path.unlink()
        parent = path.parent
        if parent.exists() and not any(parent.iterdir()):
            shutil.rmtree(parent, ignore_errors=True)
        logger.info("File deleted from disk", extra={"path": storage_path})
    except OSError as exc:
        logger.warning("Could not delete file", extra={"path": storage_path, "error": str(exc)})
