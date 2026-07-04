-- PostgreSQL initialization script
-- Runs automatically when the pgvector/pgvector:pg16 container first starts.

-- Enable pgvector extension (idempotent — safe to run multiple times)
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable gen_random_uuid() support (available by default in PG 13+)
-- CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- not needed in PG 13+
