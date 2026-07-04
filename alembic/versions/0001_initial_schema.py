"""Initial schema — documents and query_logs tables.

Creates the two custom application tables.
The document_chunks (pgvector) table is managed by LlamaIndex's PGVectorStore
and is NOT created here — it is created automatically on first use.

Revision ID: 0001
Revises:
Create Date: 2025-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── documents ──────────────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("file_type", sa.String(50), nullable=False),
        sa.Column("storage_path", sa.String(1024), nullable=False),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("chunk_count", sa.Integer, server_default="0"),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Partial index — only index documents that are NOT soft-deleted.
    op.create_index(
        "idx_documents_status",
        "documents",
        ["status"],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    # ── query_logs ────────────────────────────────────────────────────────────
    op.create_table(
        "query_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("query_text", sa.Text, nullable=False),
        sa.Column("filters", postgresql.JSONB, nullable=True),
        sa.Column("top_k", sa.Integer, nullable=True),
        sa.Column("result_chunk_ids", postgresql.JSONB, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_index(
        "idx_query_logs_created_at",
        "query_logs",
        [sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("idx_query_logs_created_at", table_name="query_logs")
    op.drop_table("query_logs")
    op.drop_index("idx_documents_status", table_name="documents")
    op.drop_table("documents")
