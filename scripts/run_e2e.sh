#!/usr/bin/env bash
# One-shot: start infra, migrate, run all query tests, write FINAL_RESULTS.txt
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Starting Postgres + Redis (docker)..."
docker compose up postgres redis -d
sleep 4

echo "==> Running migrations..."
alembic upgrade head

echo "==> Installing deps if needed..."
pip3 install -q -r requirements.txt 2>/dev/null || pip install -q -r requirements.txt

echo "==> Running all query tests..."
python3 scripts/run_all_queries_now.py

echo "==> Done. See FINAL_RESULTS.txt"
