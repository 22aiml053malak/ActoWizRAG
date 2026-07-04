"""
Alembic environment configuration.

Reads DATABASE_SYNC_URL from the environment (.env file or exported env var)
and runs migrations using the synchronous psycopg2 driver.

Usage:
    # Apply all pending migrations
    alembic upgrade head

    # Rollback one migration
    alembic downgrade -1

    # Auto-generate a new migration from ORM model changes
    alembic revision --autogenerate -m "add new column"

    # Show current migration state
    alembic current

    # Show migration history
    alembic history --verbose
"""

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool, text

from alembic import context

# Add project root to sys.path so we can import app modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env if present (development convenience).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import SQLAlchemy metadata from our ORM models.
# Alembic uses this to compare the current DB schema with the model definitions
# and auto-generate migration scripts.
from app.models.orm import Base  # noqa: E402

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Override sqlalchemy.url from the environment variable DATABASE_SYNC_URL.
# This allows us to keep the .ini file committed with a placeholder URL
# while using the real URL from .env at runtime.
database_sync_url = os.environ.get("DATABASE_SYNC_URL")
if database_sync_url:
    config.set_main_option("sqlalchemy.url", database_sync_url)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    In offline mode, Alembic generates SQL statements without connecting
    to the database. Useful for generating migration scripts to review.

    Usage: alembic upgrade head --sql > migrations.sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Include PostgreSQL-specific constructs in comparisons.
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Connects to the database and applies migrations directly.
    This is the standard mode used during deployment.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # Ensure pgvector extension exists before running migrations.
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
