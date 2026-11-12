"""Cameras: CRUD, start/stop, status, events, and a WebSocket of live events.

The API never decodes streams. It keeps the desired state in Postgres and
mirrors enabled cameras to the Redis hash ``cameras:desired``; camera shards
lease cameras from there (workers/scheduler.py) and report status to
``camera:status:<id>``. Events arrive via the event processor on the Redis
channel ``camera:<id>:events``.
"""

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status

from ..deps import check_model, require, state, ws_principal
from ..schemas import CameraIn, CameraOut, CameraPatch, EventOut
from ..services.auth import Principal

router = APIRouter(prefix="/v1/cameras", tags=["cameras"])

DESIRED_KEY = "cameras:desired"
CAMERA_MAX_KPT_VALUES = 17 * 3   # must match workers/camera_shard.py SLOT_COLS - 6
COLUMNS = "id, tenant_id, name, url, enabled, target_fps, model, config, created_at, updated_at"


def desired_entry(row) -> str:
    return json.dumps({"id": str(row["id"]), "tenant_id": str(row["tenant_id"]), "url": row["url"],
                       "target_fps": row["target_fps"], "model": row["model"], "config": json.loads(row["config"]),
                       "updated_at": row["updated_at"].isoformat()})


async def publish_desired(core, row) -> None:
    if row["enabled"]:
        await core.redis.hset(DESIRED_KEY, str(row["id"]), desired_entry(row))
    else:
        await core.redis.hdel(DESIRED_KEY, str(row["id"]))
    await core.redis.publish("cameras:changed", str(row["id"]))


async def resync_desired(core) -> int:
    """Rebuild ``cameras:desired`` from Postgres (on API start)."""
    rows = await core.pool.fetch(f"SELECT {COLUMNS} FROM camera WHERE enabled")
    pipe = core.redis.pipeline()
    pipe.delete(DESIRED_KEY)
    if rows:
        pipe.hset(DESIRED_KEY, mapping={str(r["id"]): desired_entry(r) for r in rows})
    pipe.publish("cameras:changed", "*")
    await pipe.execute()
    return len(rows)


async def _status(core, camera_id: str) -> Optional[dict]:
    raw = await core.redis.hgetall(f"camera:status:{camera_id}")
    if not raw:
        return None
    out = {}
    for k, v in raw.items():
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    out["owner"] = await core.redis.get(f"camera:lease:{camera_id}")
    return out


def _out(row, status_: Optional[dict] = None) -> CameraOut:
    return CameraOut(id=str(row["id"]), name=row["name"], url=row["url"], enabled=row["enabled"],
                     target_fps=row["target_fps"], model=row["model"], config=json.loads(row["config"]),
                     created_at=row["created_at"], updated_at=row["updated_at"], status=status_)


async def _get(core, p: Principal, camera_id: str):
    try:
        row = await core.pool.fetchrow(f"SELECT {COLUMNS} FROM camera WHERE id::text = $1 AND tenant_id::text = $2",
                                       camera_id, p.tenant_id)
    except Exception:
        row = None
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such camera")
    return row


async def _check_models(core, p: Principal, model: str, config) -> None:
    for m in [model] + config.models_used():
        check_model(p, m)
        if await core.registry.active(m) is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"model '{m}' is not serving")
    primary = await core.registry.active(model)
    # Segmentation is not offered on cameras: its mask prototypes (3.3 MB per frame) would
    # cost more than the detection itself at camera rates. Use video jobs for masks.
    if primary.task not in ("detect", "pose"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"camera model must be a detection or pose model, '{model}' is {primary.task}")
    if primary.imgsz != 640:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"camera model must take 640 px input, '{model}' takes {primary.imgsz}")
    if primary.task == "pose":
        k, d = primary.kpt_shape or (17, 3)
        if k * d > CAMERA_MAX_KPT_VALUES:
            # The shard's shared-memory output rows hold 6 box values + 17 keypoints x 3.
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"camera pose models can have at most 17 keypoints x 3, '{model}' has {k} x {d}")
    if config.shoplifting is not None:
        if primary.task != "pose":
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"shoplifting needs a pose camera model (e.g. yolo26m_pose), '{model}' is {primary.task}")
        seq = await core.registry.active(config.shoplifting.model)
        if seq.task != "sequence":
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"shoplifting.model must be a sequence model, '{config.shoplifting.model}' is {seq.task}")
    for t in config.triggered:
        trig = await core.registry.active(t.model)
        if trig.task not in ("detect", "pose"):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"triggered model '{t.model}' must be a detection or pose model, it is {trig.task}")


@router.get("", response_model=list[CameraOut])
async def list_cameras(request: Request, p: Principal = Depends(require("cameras"))):
    core = state(request)
    rows = await core.pool.fetch(f"SELECT {COLUMNS} FROM camera WHERE tenant_id::text = $1 ORDER BY name", p.tenant_id)
    return [_out(r, await _status(core, str(r["id"]))) for r in rows]


@router.post("", response_model=CameraOut, status_code=201)
async def create_camera(body: CameraIn, request: Request, p: Principal = Depends(require("cameras"))):
    core = state(request)
    await _check_models(core, p, body.model, body.config)
    async with core.pool.acquire() as conn, conn.transaction():
        limit = await conn.fetchval("SELECT max_cameras FROM tenant WHERE id::text = $1 FOR UPDATE", p.tenant_id)
        count = await conn.fetchval("SELECT count(*) FROM camera WHERE tenant_id::text = $1", p.tenant_id)
        if count >= limit:
            raise HTTPException(status.HTTP_409_CONFLICT, f"tenant camera limit ({limit}) reached")
        row = await conn.fetchrow(
            f"""INSERT INTO camera (tenant_id, name, url, enabled, target_fps, model, config)
                VALUES ($1::uuid, $2, $3, $4, $5, $6, $7) ON CONFLICT (tenant_id, name) DO NOTHING RETURNING {COLUMNS}""",
            p.tenant_id, body.name, body.url, body.enabled, body.target_fps, body.model, body.config.model_dump_json())
    if row is None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"camera '{body.name}' exists")
    await publish_desired(core, row)
    return _out(row)


@router.get("/{camera_id}", response_model=CameraOut)
async def get_camera(camera_id: str, request: Request, p: Principal = Depends(require("cameras"))):
    core = state(request)
    row = await _get(core, p, camera_id)
    return _out(row, await _status(core, camera_id))


@router.patch("/{camera_id}", response_model=CameraOut)
async def update_camera(camera_id: str, body: CameraPatch, request: Request, p: Principal = Depends(require("cameras"))):
    core = state(request)
    row = await _get(core, p, camera_id)
    model = body.model or row["model"]
    from yolo_platform_shared.camera_config import CameraConfig
    config = body.config or CameraConfig.model_validate(json.loads(row["config"]))
    await _check_models(core, p, model, config)
    row = await core.pool.fetchrow(
        f"""UPDATE camera SET name = COALESCE($2, name), url = COALESCE($3, url), target_fps = COALESCE($4, target_fps),
                model = $5, config = $6, updated_at = now() WHERE id = $1 RETURNING {COLUMNS}""",
        row["id"], body.name, body.url, body.target_fps, model, config.model_dump_json())
    await publish_desired(core, row)
    return _out(row)


@router.delete("/{camera_id}", status_code=204)
async def delete_camera(camera_id: str, request: Request, p: Principal = Depends(require("cameras"))):
    core = state(request)
    row = await _get(core, p, camera_id)
    await core.pool.execute("DELETE FROM camera WHERE id = $1", row["id"])
    await core.redis.hdel(DESIRED_KEY, camera_id)
    await core.redis.publish("cameras:changed", camera_id)


async def _set_enabled(request: Request, p: Principal, camera_id: str, enabled: bool) -> CameraOut:
    core = state(request)
    row = await _get(core, p, camera_id)
    if enabled:
        from yolo_platform_shared.camera_config import CameraConfig
        await _check_models(core, p, row["model"], CameraConfig.model_validate(json.loads(row["config"])))
    row = await core.pool.fetchrow(
        f"UPDATE camera SET enabled = $2, updated_at = now() WHERE id = $1 RETURNING {COLUMNS}", row["id"], enabled)
    await publish_desired(core, row)
    return _out(row)


@router.post("/{camera_id}/start", response_model=CameraOut)
async def start_camera(camera_id: str, request: Request, p: Principal = Depends(require("cameras"))):
    return await _set_enabled(request, p, camera_id, True)


@router.post("/{camera_id}/stop", response_model=CameraOut)
async def stop_camera(camera_id: str, request: Request, p: Principal = Depends(require("cameras"))):
    return await _set_enabled(request, p, camera_id, False)


@router.get("/{camera_id}/events", response_model=list[EventOut])
async def camera_events(camera_id: str, request: Request, limit: int = Query(100, ge=1, le=1000),
                        before_id: Optional[int] = None, p: Principal = Depends(require("cameras"))):
    core = state(request)
    row = await _get(core, p, camera_id)
    rows = await core.pool.fetch(
        """SELECT id, camera_id, ts, type, rule, track_id, class_name, confidence, bbox, data FROM event
           WHERE camera_id = $1 AND ($2::bigint IS NULL OR id < $2) ORDER BY id DESC LIMIT $3""",
        row["id"], before_id, limit)
    return [EventOut(id=r["id"], camera_id=str(r["camera_id"]), ts=r["ts"], type=r["type"], rule=r["rule"],
                     track_id=r["track_id"], class_name=r["class_name"], confidence=r["confidence"],
                     bbox=json.loads(r["bbox"]) if r["bbox"] else None, data=json.loads(r["data"])) for r in rows]


@router.websocket("/{camera_id}/stream")
async def camera_stream(ws: WebSocket, camera_id: str):
    """Live events for one camera, plus a status message every 2 s. Auth: X-API-Key header or ?api_key=."""
    p = await ws_principal(ws, "cameras")
    if p is None:
        await ws.close(code=4401)
        return
    core = ws.app.state.core
    try:
        await _get(core, p, camera_id)
    except HTTPException:
        await ws.close(code=4404)
        return
    await ws.accept()
    pubsub = core.redis.pubsub()
    await pubsub.subscribe(f"camera:{camera_id}:events")
    try:
        loop = asyncio.get_running_loop()
        next_status = 0.0
        while True:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg is not None:
                await ws.send_text(json.dumps({"type": "event", "event": json.loads(msg["data"])}))
            if loop.time() >= next_status:
                await ws.send_text(json.dumps({"type": "status", "status": await _status(core, camera_id)}, default=str))
                next_status = loop.time() + 2.0
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await pubsub.unsubscribe()
        await pubsub.aclose()
