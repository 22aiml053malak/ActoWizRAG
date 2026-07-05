import os

import sqlalchemy as sa
from dotenv import load_dotenv

from app.repositories.vector_repository import PHYSICAL_CHUNK_TABLE_NAME

load_dotenv()
db_url = os.getenv(
    "DATABASE_SYNC_URL",
    "postgresql://postgres:postgres@127.0.0.1:5432/actowiz_rag",
)
engine = sa.create_engine(db_url)
try:
    count = engine.connect().execute(
        sa.text(f"SELECT COUNT(*) FROM {PHYSICAL_CHUNK_TABLE_NAME}")
    ).scalar()
    print(count)
except Exception as e:
    print("Error:", e)
finally:
    engine.dispose()
