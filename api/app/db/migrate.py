"""Apply database/migrations/NNNN_*.sql in order, each once, under an advisory lock.

Run by the one-shot ``migrate`` compose service before the API starts:

    python -m app.db.migrate
"""

import asyncio
import logging
import os
import re
import sys
from pathlib import Path

import asyncpg

logger = logging.getLogger("migrate")
LOCK_ID = 7_202_611  # arbitrary, constant: serialises concurrent migrators
NAME = re.compile(r"^(\d{4})_[\w-]+\.sql$")


async def migrate(dsn: str, directory: Path) -> list[str]:
    conn = await asyncpg.connect(dsn)
    applied_now = []
    try:
        await conn.execute("SELECT pg_advisory_lock($1)", LOCK_ID)
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migration (version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {r["version"] for r in await conn.fetch("SELECT version FROM schema_migration")}
        for path in sorted(directory.iterdir()):
            m = NAME.match(path.name)
            if not m or path.stem in done:
                continue
            async with conn.transaction():
                await conn.execute(path.read_text())
                await conn.execute("INSERT INTO schema_migration (version) VALUES ($1)", path.stem)
            applied_now.append(path.stem)
            logger.info("applied %s", path.name)
    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", LOCK_ID)
        await conn.close()
    return applied_now


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    dsn = os.environ["DATABASE_URL"]
    directory = Path(os.environ.get("MIGRATIONS_DIR", "/app/migrations"))
    applied = asyncio.run(migrate(dsn, directory))
    logger.info("schema up to date (%d applied now)", len(applied))
    return 0


if __name__ == "__main__":
    sys.exit(main())
