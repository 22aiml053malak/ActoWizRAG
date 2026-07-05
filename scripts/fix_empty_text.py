"""
One-time fix: backfill the `text` column in data_document_chunks for rows
where text is empty but metadata_->>'original_text' has content.

Run from the project root:
    python scripts/fix_empty_text.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.repositories.vector_repository import VectorRepository

if __name__ == "__main__":
    repo = VectorRepository()
    updated = repo.backfill_empty_text()
    print(f"Done. {updated} rows updated.")
