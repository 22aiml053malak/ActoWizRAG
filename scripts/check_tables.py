import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sqlalchemy as sa
from app.core.config import settings

engine = sa.create_engine(settings.DATABASE_SYNC_URL)
with engine.connect() as conn:
    rows = conn.execute(sa.text(
        "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
    )).fetchall()
    print("Tables:", [r[0] for r in rows])

    # check document table columns
    for t in [r[0] for r in rows]:
        if 'doc' in t.lower():
            cols = conn.execute(sa.text(
                f"SELECT column_name FROM information_schema.columns WHERE table_name='{t}'"
            )).fetchall()
            print(f"\n{t} columns:", [c[0] for c in cols])
            sample = conn.execute(sa.text(f"SELECT * FROM {t} LIMIT 1")).fetchone()
            print(f"{t} sample:", sample)
engine.dispose()
