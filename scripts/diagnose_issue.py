"""
Diagnostic script to check database state and identify query issues.
"""
import os
import sys

import sqlalchemy as sa
from dotenv import load_dotenv

from app.repositories.vector_repository import (
    CHUNK_TABLE_NAME,
    PHYSICAL_CHUNK_TABLE_NAME,
)

load_dotenv()
db_url = os.getenv(
    "DATABASE_SYNC_URL",
    "postgresql://postgres:postgres@127.0.0.1:5432/actowiz_rag",
)

print(f"Connecting to: {db_url}")
print(f"PGVectorStore logical table: {CHUNK_TABLE_NAME}")
print(f"PGVectorStore physical table: {PHYSICAL_CHUNK_TABLE_NAME}")
print("=" * 80)

engine = sa.create_engine(db_url)

try:
    with engine.connect() as conn:
        print("✓ Database connection successful\n")

        result = conn.execute(
            sa.text(
                "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
            )
        )
        ext = result.fetchone()
        if ext:
            print(f"✓ pgvector extension installed (version {ext[1]})\n")
        else:
            print("✗ pgvector extension NOT installed\n")

        app_tables = ["documents", "query_logs"]
        print("Application tables:")
        for table in app_tables:
            result = conn.execute(
                sa.text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = :table)"
                ),
                {"table": table},
            )
            exists = result.scalar()
            status = "✓" if exists else "✗"
            print(f"  {status} {table}")

        print("\nVector chunk tables:")
        for table in [CHUNK_TABLE_NAME, PHYSICAL_CHUNK_TABLE_NAME]:
            result = conn.execute(
                sa.text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = :table)"
                ),
                {"table": table},
            )
            exists = result.scalar()
            status = "✓" if exists else "✗"
            print(f"  {status} {table}")

        print("\nRow counts:")
        for table in app_tables + [PHYSICAL_CHUNK_TABLE_NAME]:
            try:
                result = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                print(f"  {table}: {count} rows")
            except Exception as e:
                print(f"  {table}: ERROR - {e}")

        print()
        try:
            result = conn.execute(
                sa.text("SELECT status, COUNT(*) FROM documents GROUP BY status")
            )
            print("Documents by status:")
            for row in result:
                print(f"  {row[0]}: {row[1]}")
        except Exception as e:
            print(f"  ERROR: {e}")

        print()
        try:
            result = conn.execute(
                sa.text(
                    f"SELECT id, metadata_->>'document_id' as doc_id, "
                    f"metadata_->>'filename' as filename, "
                    f"LEFT(text, 100) as text_preview "
                    f"FROM {PHYSICAL_CHUNK_TABLE_NAME} LIMIT 3"
                )
            )
            print(f"Sample chunks from {PHYSICAL_CHUNK_TABLE_NAME}:")
            rows = result.fetchall()
            if rows:
                for row in rows:
                    print(f"  - Chunk ID: {row[0][:8]}...")
                    print(f"    Doc ID: {row[1]}")
                    print(f"    Filename: {row[2]}")
                    print(f"    Text: {row[3]}...")
                    print()
            else:
                print("  No chunks found — upload a document and wait for ingestion")
        except Exception as e:
            print(f"  ERROR: {e}")

        print()
        try:
            result = conn.execute(
                sa.text(
                    f"SELECT COUNT(*) FROM {PHYSICAL_CHUNK_TABLE_NAME} "
                    "WHERE embedding IS NOT NULL"
                )
            )
            count = result.scalar()
            print(f"Chunks with embeddings: {count}")
        except Exception as e:
            print(f"ERROR checking embeddings: {e}")

except sa.exc.OperationalError as e:
    print("✗ Database connection failed:")
    print(f"  {e}")
    print("\nPossible issues:")
    print("  1. Database doesn't exist - run: createdb actowiz_rag")
    print("  2. Postgres not running - start postgres or: docker compose up postgres -d")
    print("  3. Wrong credentials in .env (docker-compose uses postgres/postgres)")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)
finally:
    engine.dispose()

print("\n" + "=" * 80)
print("Diagnosis complete")
