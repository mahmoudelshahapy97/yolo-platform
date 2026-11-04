"""Model registry: which model versions exist, which one is active, and rollouts.

The source of truth for *what exists* is the Triton model repository on disk
(<repo>/<name>/<version>/model.plan + model_meta.json). The source of truth for
*which version serves* is Postgres (model_version.status = 'active'), mirrored to
the Redis hash ``models:active`` so camera and video workers pick the same
version without a database round trip.

Activating a version (first boot, rollout or rollback):
  1. load the candidate next to the current version (explicit model control),
  2. run the gate: the candidate's outputs on its golden.npz must match the
     reference recorded from the .pt (shared/yolo_platform_shared/gate.py),
  3. if it passes: mark it active and publish it; the old version stays loaded
     for ``drain_s`` seconds so in-flight requests finish, then is unloaded,
  4. if it fails: mark it failed and unload it; the current version keeps serving.
Every rollout for a model holds a Redis lock, so two API processes never race.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import asyncpg
import numpy as np
from redis.asyncio import Redis

from yolo_platform_shared.gate import evaluate_classify, evaluate_detect

from .triton_client import TritonClient

logger = logging.getLogger(__name__)

ACTIVE_KEY = "models:active"
CHANNEL = "models:changed"


@dataclass
class ActiveModel:
    name: str
    version: int
    task: str
    imgsz: int
    max_batch: int
    names: Dict[int, str]


class GateFailed(Exception):
    def __init__(self, report: dict) -> None:
        super().__init__("gate failed")
        self.report = report


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelRegistry:
    def __init__(self, pool: asyncpg.Pool, redis: Redis, triton: TritonClient, repository: Path, drain_s: float = 10.0):
        self.pool = pool
        self.redis = redis
        self.triton = triton
        self.repository = repository
        self.drain_s = drain_s
        self._cache: Dict[str, ActiveModel] = {}
        self._cache_at = 0.0

    # ---------- discovery ----------

    def _meta(self, name: str, version: int) -> Optional[dict]:
        for p in (self.repository / name / str(version) / "model_meta.json", self.repository / name / "model_meta.json"):
            if p.is_file():
                return json.loads(p.read_text())
        return None

    def repository_versions(self) -> Dict[str, list]:
        found: Dict[str, list] = {}
        if not self.repository.is_dir():
            return found
        for model_dir in sorted(self.repository.iterdir()):
            if not (model_dir / "config.pbtxt").is_file():
                continue
            versions = sorted(int(v.name) for v in model_dir.iterdir() if v.is_dir() and v.name.isdigit()
                              and (v / "model.plan").is_file() and (v / "model.plan").stat().st_size > 0)
            if versions:
                found[model_dir.name] = versions
        return found

    async def sync(self) -> int:
        """Register every version on disk that the database does not know yet."""
        added = 0
        for name, versions in self.repository_versions().items():
            for v in versions:
                meta = self._meta(name, v)
                if not meta:
                    logger.warning("%s v%s has no model_meta.json; skipped", name, v)
                    continue
                sha_file = self.repository / name / str(v) / "model.plan.sha256"
                sha = sha_file.read_text().strip() if sha_file.is_file() else None
                status = await self.pool.fetchval(
                    """INSERT INTO model_version (model_name, version, task, imgsz, max_batch, names, sha256)
                       VALUES ($1, $2, $3, $4, $5, $6, $7)
                       ON CONFLICT (model_name, version) DO NOTHING RETURNING status""",
                    name, v, meta["task"], meta["imgsz"], meta["max_batch"], json.dumps(meta["names"]), sha,
                )
                added += status is not None
        return added

    # ---------- reads ----------

    async def list_versions(self, name: Optional[str] = None) -> list:
        rows = await self.pool.fetch(
            """SELECT model_name, version, task, framework, runtime, precision, imgsz, max_batch, sha256, status,
                      gate_report, created_at, activated_at, (SELECT count(*) FROM jsonb_object_keys(names)) AS num_classes
               FROM model_version WHERE ($1::text IS NULL OR model_name = $1) ORDER BY model_name, version""",
            name,
        )
        return [dict(r) | {"gate_report": json.loads(r["gate_report"]) if r["gate_report"] else None} for r in rows]

    async def active(self, name: str) -> Optional[ActiveModel]:
        now = time.monotonic()
        if now - self._cache_at > 2.0:
            await self._refresh_cache()
        return self._cache.get(name)

    async def all_active(self) -> Dict[str, ActiveModel]:
        await self._refresh_cache()
        return dict(self._cache)

    async def _refresh_cache(self) -> None:
        rows = await self.pool.fetch(
            "SELECT model_name, version, task, imgsz, max_batch, names FROM model_version WHERE status = 'active'"
        )
        self._cache = {
            r["model_name"]: ActiveModel(r["model_name"], r["version"], r["task"], r["imgsz"], r["max_batch"],
                                         {int(k): v for k, v in json.loads(r["names"]).items()})
            for r in rows
        }
        self._cache_at = time.monotonic()

    # ---------- rollout ----------

    async def gate(self, name: str, version: int) -> dict:
        golden_path = self.repository / name / str(version) / "golden.npz"
        if not golden_path.is_file():
            return {"passed": False, "error": f"no golden.npz for {name} v{version}"}
        meta = self._meta(name, version)
        sha_file = self.repository / name / str(version) / "model.plan.sha256"
        if sha_file.is_file():
            actual = await asyncio.to_thread(file_sha256, self.repository / name / str(version) / "model.plan")
            if actual != sha_file.read_text().strip():
                return {"passed": False, "error": "model.plan sha256 does not match model.plan.sha256"}
        golden = dict(np.load(golden_path))
        inputs = golden["inputs"]
        outs = []
        for start in range(0, len(inputs), meta["max_batch"]):
            outs.append(await self.triton.infer(name, version, inputs[start:start + meta["max_batch"]]))
        outputs = np.concatenate(outs)
        evaluate = evaluate_classify if meta["task"] == "classify" else evaluate_detect
        return await asyncio.to_thread(evaluate, golden, outputs)

    async def _wait_ready(self, name: str, version: int, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self.triton.model_ready(name, str(version)):
                return
            await asyncio.sleep(0.5)
        raise TimeoutError(f"{name} v{version} not ready after {timeout}s")

    async def activate(self, name: str, version: int, reason: str = "rollout") -> dict:
        lock = self.redis.lock(f"lock:model:{name}", timeout=600, blocking_timeout=600)
        async with lock:
            current = await self.pool.fetchval(
                "SELECT version FROM model_version WHERE model_name = $1 AND status = 'active'", name)
            row = await self.pool.fetchrow(
                "SELECT status FROM model_version WHERE model_name = $1 AND version = $2", name, version)
            if row is None:
                raise LookupError(f"{name} v{version} is not registered")
            if current == version:
                await self.publish()
                return {"model": name, "version": version, "status": "active", "changed": False}
            config_text = (self.repository / name / "config.pbtxt").read_text()
            keep = [current] if current is not None else []
            await self.triton.load(name, keep + [version], config_text)
            await self._wait_ready(name, version)
            report = await self.gate(name, version)
            report["reason"] = reason
            if not report.get("passed"):
                await self.pool.execute(
                    "UPDATE model_version SET status = 'failed', gate_report = $3 WHERE model_name = $1 AND version = $2",
                    name, version, json.dumps(report))
                if keep:
                    await self.triton.load(name, keep, config_text)
                else:
                    await self.triton.unload(name)
                raise GateFailed(report)
            async with self.pool.acquire() as conn, conn.transaction():
                await conn.execute(
                    "UPDATE model_version SET status = 'retired' WHERE model_name = $1 AND status = 'active'", name)
                await conn.execute(
                    """UPDATE model_version SET status = 'active', gate_report = $3, activated_at = now()
                       WHERE model_name = $1 AND version = $2""", name, version, json.dumps(report))
            self._cache_at = 0.0
            await self.publish()
            logger.info("%s: v%s active (was v%s, %s)", name, version, current, reason)
        if keep:
            asyncio.create_task(self._drain_and_unload(name, version, config_text))
        return {"model": name, "version": version, "previous": current, "status": "active", "changed": True, "gate": report}

    async def _drain_and_unload(self, name: str, version: int, config_text: str) -> None:
        await asyncio.sleep(self.drain_s)
        async with self.redis.lock(f"lock:model:{name}", timeout=600, blocking_timeout=600):
            still = await self.pool.fetchval(
                "SELECT version FROM model_version WHERE model_name = $1 AND status = 'active'", name)
            if still == version:
                await self.triton.load(name, [version], config_text)

    async def rollback(self, name: str) -> dict:
        previous = await self.pool.fetchval(
            """SELECT version FROM model_version WHERE model_name = $1 AND status = 'retired' AND activated_at IS NOT NULL
               ORDER BY activated_at DESC LIMIT 1""", name)
        if previous is None:
            raise LookupError(f"{name} has no previous version to roll back to")
        return await self.activate(name, previous, reason="rollback")

    async def publish(self) -> None:
        """Mirror the active versions to Redis for the workers."""
        rows = await self.pool.fetch(
            "SELECT model_name, version, task, imgsz, max_batch, names FROM model_version WHERE status = 'active'")
        mapping = {r["model_name"]: json.dumps({"version": r["version"], "task": r["task"], "imgsz": r["imgsz"],
                                                 "max_batch": r["max_batch"], "names": json.loads(r["names"])})
                   for r in rows}
        pipe = self.redis.pipeline()
        pipe.delete(ACTIVE_KEY)
        if mapping:
            pipe.hset(ACTIVE_KEY, mapping=mapping)
        pipe.publish(CHANNEL, "1")
        await pipe.execute()

    async def bootstrap(self) -> None:
        """On start: register new versions, load each active version, activate the newest for models with none."""
        async with self.redis.lock("lock:models:bootstrap", timeout=900, blocking_timeout=900):
            await self.sync()
            on_disk = self.repository_versions()
            active = {r["model_name"]: r["version"] for r in await self.pool.fetch(
                "SELECT model_name, version FROM model_version WHERE status = 'active'")}
            for name, versions in on_disk.items():
                config_text = (self.repository / name / "config.pbtxt").read_text()
                if name in active and active[name] in versions:
                    if not await self.triton.model_ready(name, str(active[name])):
                        await self.triton.load(name, [active[name]], config_text)
                    continue
                candidates = [v for v in versions if await self._status(name, v) not in ("failed",)]
                if not candidates:
                    logger.error("%s: every version failed its gate; not serving", name)
                    continue
                try:
                    await self.activate(name, max(candidates), reason="bootstrap")
                except GateFailed as exc:
                    logger.error("%s v%s failed its gate on bootstrap: %s", name, max(candidates), exc.report)
            await self.publish()

    async def _status(self, name: str, version: int) -> Optional[str]:
        return await self.pool.fetchval(
            "SELECT status FROM model_version WHERE model_name = $1 AND version = $2", name, version)
