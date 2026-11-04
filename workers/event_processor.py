"""Event processor: Redis Stream ``events`` -> Postgres, plus live fan-out.

    python -m workers.event_processor

Consumer group ``event-processor``: entries are acknowledged only after the
insert commits, so a crash redelivers them. The stream entry id is stored with
a unique constraint, so a redelivered event is written once. Entries left pending
by a dead consumer are reclaimed with XAUTOCLAIM. Each stored event is
published on ``camera:<id>:events`` for the API's WebSocket.
"""

import asyncio
import json
import logging
import os
import signal
import socket
from datetime import datetime, timezone

import asyncpg
from prometheus_client import Counter, start_http_server
from redis.asyncio import Redis
from redis.exceptions import ResponseError

logger = logging.getLogger("event-processor")

STREAM, GROUP = "events", "event-processor"
M_STORED = Counter("events_stored_total", "Events written to Postgres")
M_DUP = Counter("events_duplicate_total", "Redelivered events already stored")
M_FAIL = Counter("events_failed_total", "Events that could not be stored (left pending)")
M_ORPHAN = Counter("events_orphan_total", "Events for cameras that no longer exist (acknowledged, dropped)")

INSERT = """
INSERT INTO event (camera_id, ts, type, rule, track_id, class_name, confidence, bbox, data, stream_id)
VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (stream_id) DO NOTHING RETURNING id
"""


async def store(pool: asyncpg.Pool, redis: Redis, entries) -> list:
    acked = []
    for entry_id, fields in entries:
        try:
            ev = json.loads(fields["event"])
            row_id = await pool.fetchval(
                INSERT, ev["camera_id"], datetime.fromtimestamp(ev["ts"], tz=timezone.utc), ev["type"], ev["rule"],
                ev.get("track_id"), ev.get("class_name"), ev.get("confidence"),
                json.dumps(ev["bbox"]) if ev.get("bbox") else None, json.dumps(ev.get("data") or {}), entry_id)
        except asyncpg.ForeignKeyViolationError:
            M_ORPHAN.inc()
            acked.append(entry_id)
            continue
        except (KeyError, ValueError, TypeError) as exc:
            logger.error("malformed event %s dropped: %s", entry_id, exc)
            acked.append(entry_id)
            continue
        except Exception:
            M_FAIL.inc()
            logger.exception("could not store event %s; it stays pending", entry_id)
            continue
        acked.append(entry_id)
        if row_id is None:
            M_DUP.inc()
            continue
        M_STORED.inc()
        ev["id"] = row_id
        await redis.publish(f"camera:{ev['camera_id']}:events", json.dumps(ev))
    return acked


async def run(stop: asyncio.Event) -> None:
    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=1, max_size=4)
    redis = Redis.from_url(os.environ.get("REDIS_URL", "redis://redis:6379/0"), decode_responses=True)
    consumer = f"{socket.gethostname()}-{os.getpid()}"
    try:
        await redis.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise
    logger.info("consuming %s as %s", STREAM, consumer)
    next_claim = 0.0
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        try:
            if loop.time() >= next_claim:
                # Take over entries another consumer read but never acknowledged (it died).
                _, claimed, _ = await redis.xautoclaim(STREAM, GROUP, consumer, min_idle_time=60_000, count=500)
                if claimed:
                    acked = await store(pool, redis, claimed)
                    if acked:
                        await redis.xack(STREAM, GROUP, *acked)
                next_claim = loop.time() + 30
            resp = await redis.xreadgroup(GROUP, consumer, {STREAM: ">"}, count=500, block=1000)
            for _, entries in resp or []:
                acked = await store(pool, redis, entries)
                if acked:
                    await redis.xack(STREAM, GROUP, *acked)
        except Exception:
            logger.exception("event loop error")
            await asyncio.sleep(1)
    await redis.aclose()
    await pool.close()


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    start_http_server(int(os.environ.get("METRICS_PORT", "9100")))
    stop = asyncio.Event()

    async def _main():
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        await run(stop)

    asyncio.run(_main())


if __name__ == "__main__":
    main()
