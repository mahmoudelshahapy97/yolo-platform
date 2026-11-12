"""Model registry: which model versions exist, which one is active, and rollouts.

The source of truth for *what exists* is the model repository on disk
(<repo>/<name>/<version>/model.plan + model_meta.json + golden.npz). The source of
truth for *which version serves* is Postgres (model_version.status = 'active'),
mirrored to the Redis hash ``models:active`` so camera and video workers use the
same version without a database round trip.

Every registry version is served as its own Triton model, ``<name>__v<version>``,
in a generated serving repository (config derived from <repo>/<name>/config.pbtxt,
plan symlinked from <repo>/<name>/<version>/). A model that is serving is never
reloaded: with Triton 25.10, re-loading a detector (embedded NMS) while one of its
versions is kept made the next request to the kept version segfault the server.

Activating a version (first boot, rollout or rollback):
  1. load ``<name>__v<new>`` next to the serving model (explicit model control),
  2. run the gate: its outputs on golden.npz must match the reference recorded
     from the .pt (shared/yolo_platform_shared/gate.py), and the plan's sha256 must
     match,
  3. if it passes: mark it active and publish it; the old model stays loaded for
     ``drain_s`` seconds so in-flight requests finish, then is unloaded,
  4. if it fails: mark it failed and unload it; the current version keeps serving.
Every rollout for a model holds a Redis lock, so two API processes never race.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import asyncpg
import numpy as np
from google.protobuf import text_format
from redis.asyncio import Redis
from tritonclient.grpc import model_config_pb2

from yolo_platform_shared.gate import evaluate

from .triton_client import TritonClient

logger = logging.getLogger(__name__)

ACTIVE_KEY = "models:active"
CHANNEL = "models:changed"


def serve_name(name: str, version: int) -> str:
    """Triton model name serving registry version ``version`` of ``name``."""
    return f"{name}__v{version}"


IMAGE_INPUT = {"name": "images", "dtype": "UINT8"}


@dataclass
class ActiveModel:
    name: str
    version: int
    task: str
    imgsz: int
    max_batch: int
    names: Dict[int, str]
    # From model_meta.json: the engine's input and outputs (config order), pose keypoint layout.
    input: Optional[dict] = None
    outputs: Tuple[str, ...] = ("output0",)
    kpt_shape: Optional[Tuple[int, int]] = None

    @property
    def serve_name(self) -> str:
        return serve_name(self.name, self.version)

    @property
    def input_name(self) -> str:
        return (self.input or IMAGE_INPUT)["name"]


def io_fields(meta: Optional[dict]) -> dict:
    """The I/O description workers and adapters need, from a version's model_meta.json."""
    meta = meta or {}
    return {
        "input": meta.get("input") or {**IMAGE_INPUT, "dims": [meta.get("imgsz", 0), meta.get("imgsz", 0), 3]},
        "outputs": [o["name"] for o in meta.get("outputs", [{"name": "output0"}])],
        "kpt_shape": meta.get("kpt_shape"),
    }


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
    def __init__(self, pool: asyncpg.Pool, redis: Redis, triton: TritonClient, repository: Path, serving: Path,
                 drain_s: float = 10.0, triton_models_path: str = "/models"):
        self.pool = pool
        self.redis = redis
        self.triton = triton
        self.repository = repository
        self.serving = serving
        # Where Triton sees the source repository (the serving symlinks point there).
        self.triton_models_path = triton_models_path.rstrip("/")
        self.drain_s = drain_s
        self._cache: Dict[str, ActiveModel] = {}
        self._cache_at = 0.0
        # A version's model_meta.json does not change once registered: read it once, off the loop.
        self._io: Dict[Tuple[str, int], dict] = {}

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
                    """INSERT INTO model_version (model_name, version, task, imgsz, max_batch, names, sha256,
                                                  framework, precision)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                       ON CONFLICT (model_name, version) DO NOTHING RETURNING status""",
                    name, v, meta["task"], meta["imgsz"], meta["max_batch"], json.dumps(meta["names"]), sha,
                    meta.get("framework", "ultralytics"), meta.get("precision", "fp16"),
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
        cache = {}
        for r in rows:
            io = await self._io_fields(r["model_name"], r["version"])
            cache[r["model_name"]] = ActiveModel(
                r["model_name"], r["version"], r["task"], r["imgsz"], r["max_batch"],
                {int(k): v for k, v in json.loads(r["names"]).items()}, input=io["input"],
                outputs=tuple(io["outputs"]), kpt_shape=tuple(io["kpt_shape"]) if io["kpt_shape"] else None)
        self._cache = cache
        self._cache_at = time.monotonic()

    async def _io_fields(self, name: str, version: int) -> dict:
        key = (name, version)
        io = self._io.get(key)
        if io is None:
            meta = await asyncio.to_thread(self._meta, name, version)
            io = io_fields(meta)
            if meta is not None:   # a missing file may still appear: only cache what was read
                self._io[key] = io
        return io

    # ---------- serving repository ----------

    def materialize(self, name: str, version: int) -> str:
        """Write serving/<name>__v<version>/ (config + symlinked plan); returns the Triton model name."""
        sname = serve_name(name, version)
        src_cfg = (self.repository / name / "config.pbtxt").read_text()
        cfg = text_format.Parse(src_cfg, model_config_pb2.ModelConfig())
        cfg.name = sname
        cfg.ClearField("version_policy")  # one version (1) per serving model: Triton default
        if cfg.output and cfg.output[0].label_filename:
            cfg.output[0].label_filename = "labels.txt"
        target = self.serving / sname
        tmp = self.serving / f".{sname}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "1").mkdir(parents=True)
        (tmp / "config.pbtxt").write_text(text_format.MessageToString(cfg))
        os.symlink(f"{self.triton_models_path}/{name}/{version}/model.plan", tmp / "1" / "model.plan")
        if (self.repository / name / "labels.txt").is_file():
            os.symlink(f"{self.triton_models_path}/{name}/labels.txt", tmp / "labels.txt")
        shutil.rmtree(target, ignore_errors=True)
        tmp.rename(target)
        return sname

    async def _loaded(self) -> set:
        """Triton model names currently READY."""
        try:
            index = await self.triton.repository_index()
        except Exception:
            return set()
        return {m["name"] for m in index if m.get("state") == "READY"}

    async def _ensure_loaded(self, name: str, version: int) -> str:
        sname = serve_name(name, version)
        if sname not in await self._loaded():
            await asyncio.to_thread(self.materialize, name, version)
            await self.triton.load(sname)
        await self._wait_ready(sname)
        return sname

    async def _wait_ready(self, sname: str, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self.triton.model_ready(sname):
                return
            await asyncio.sleep(0.5)
        raise TimeoutError(f"{sname} not ready after {timeout}s")

    async def _unload(self, name: str, version: int) -> None:
        sname = serve_name(name, version)
        try:
            await self.triton.unload(sname)
        except Exception as exc:
            logger.warning("unload %s: %s", sname, exc)

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
        io = io_fields(meta)
        chunks: List[List[np.ndarray]] = []
        for start in range(0, len(inputs), meta["max_batch"]):
            chunks.append(await self.triton.infer_many(serve_name(name, version), 1,
                                                       inputs[start:start + meta["max_batch"]], io["outputs"],
                                                       io["input"]["name"]))
        outputs = [np.concatenate([c[i] for c in chunks]) for i in range(len(io["outputs"]))]
        return await asyncio.to_thread(evaluate, meta["task"], golden, outputs, meta)

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
                await self._ensure_loaded(name, version)
                await self.publish()
                return {"model": name, "version": version, "status": "active", "changed": False}
            await self._ensure_loaded(name, version)
            report = await self.gate(name, version)
            report["reason"] = reason
            if not report.get("passed"):
                await self.pool.execute(
                    "UPDATE model_version SET status = 'failed', gate_report = $3 WHERE model_name = $1 AND version = $2",
                    name, version, json.dumps(report))
                await self._unload(name, version)
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
        if current is not None:
            asyncio.create_task(self._drain_and_unload(name, current))
        return {"model": name, "version": version, "previous": current, "status": "active", "changed": True,
                "gate": report}

    async def _drain_and_unload(self, name: str, old_version: int) -> None:
        """Unload the previous version once in-flight requests (and workers' version caches) moved on."""
        await asyncio.sleep(self.drain_s)
        async with self.redis.lock(f"lock:model:{name}", timeout=600, blocking_timeout=600):
            still = await self.pool.fetchval(
                "SELECT version FROM model_version WHERE model_name = $1 AND status = 'active'", name)
            if still != old_version:
                await self._unload(name, old_version)

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
        mapping = {r["model_name"]: json.dumps({"version": r["version"], "serve_name": serve_name(r["model_name"], r["version"]),
                                                 "task": r["task"], "imgsz": r["imgsz"], "max_batch": r["max_batch"],
                                                 "names": json.loads(r["names"]),
                                                 **io_fields(self._meta(r["model_name"], r["version"]))})
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
            self.serving.mkdir(parents=True, exist_ok=True)
            await self.sync()
            on_disk = self.repository_versions()
            active = {r["model_name"]: r["version"] for r in await self.pool.fetch(
                "SELECT model_name, version FROM model_version WHERE status = 'active'")}
            for name, versions in on_disk.items():
                if name in active and active[name] in versions:
                    await self._ensure_loaded(name, active[name])
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

    async def watchdog(self, every_s: float = 10.0) -> None:
        """Reload active versions Triton no longer serves (a Triton restart comes back with
        nothing loaded in explicit model-control mode), and unload strays (left by a crash
        mid-rollout). Skips any model whose rollout lock is held."""
        while True:
            await asyncio.sleep(every_s)
            try:
                if not await self.triton.live():
                    continue
                active = {r["model_name"]: r["version"] for r in await self.pool.fetch(
                    "SELECT model_name, version FROM model_version WHERE status = 'active'")}
                loaded = await self._loaded()
                wanted = {serve_name(n, v) for n, v in active.items()}
                names = set(active) | {s.split("__v")[0] for s in loaded if "__v" in s}
                for name in names:
                    lock = self.redis.lock(f"lock:model:{name}", timeout=600, blocking_timeout=0.5)
                    if not await lock.acquire():
                        continue  # a rollout (or another API process) is on it
                    try:
                        loaded = await self._loaded()
                        if name in active and serve_name(name, active[name]) not in loaded:
                            logger.warning("%s v%s is not loaded in Triton; reloading", name, active[name])
                            await self._ensure_loaded(name, active[name])
                        for stray in [s for s in loaded if s.startswith(f"{name}__v") and s not in wanted]:
                            logger.warning("unloading stray %s", stray)
                            await self.triton.unload(stray)
                    finally:
                        await lock.release()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("model watchdog")

    async def _status(self, name: str, version: int) -> Optional[str]:
        return await self.pool.fetchval(
            "SELECT status FROM model_version WHERE model_name = $1 AND version = $2", name, version)
