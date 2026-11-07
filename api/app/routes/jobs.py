"""Asynchronous video jobs: upload -> object storage -> job row -> Redis Stream -> video worker."""

import asyncio
import json
import tempfile
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, \
    WebSocketDisconnect, status

from .. import metrics
from ..deps import check_model, require, state, ws_principal
from ..schemas import JobOut
from ..services.auth import Principal

router = APIRouter(tags=["jobs"])

JOB_STREAM = "jobs:video"
VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-matroska", "video/webm", "video/x-msvideo",
               "application/octet-stream"}
JOB_COLUMNS = ("id, status, model, params, total_frames, processed_frames, attempts, error, summary, created_at, "
               "started_at, finished_at, result_key, output_key")


async def _job_out(core, row) -> JobOut:
    total = row["total_frames"]
    progress = 100.0 if row["status"] == "completed" else (
        round(100.0 * row["processed_frames"] / total, 1) if total else 0.0)
    return JobOut(
        job_id=str(row["id"]), status=row["status"], model=row["model"], params=json.loads(row["params"]),
        progress=min(progress, 100.0), processed_frames=row["processed_frames"], total_frames=total,
        attempts=row["attempts"], error=row["error"], summary=json.loads(row["summary"]) if row["summary"] else None,
        created_at=row["created_at"], started_at=row["started_at"], finished_at=row["finished_at"],
        result_url=await core.storage.presign(row["result_key"]) if row["result_key"] else None,
        output_url=await core.storage.presign(row["output_key"]) if row["output_key"] else None,
    )


@router.post("/v1/infer/videos", status_code=202)
async def submit_video(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("yolo26n_detection"),
    sample_fps: float = Form(5.0, gt=0, le=60),
    conf: float = Form(0.25, ge=0.01, le=1.0),
    annotate: bool = Form(False, description="also render an annotated MP4"),
    track: bool = Form(True, description="assign track ids (ByteTrack)"),
    p: Principal = Depends(require("jobs")),
):
    core = state(request)
    check_model(p, model)
    active = await core.registry.active(model)
    if active is None or active.task not in ("detect", "pose", "segment"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"'{model}' is not a serving detection, pose or segmentation model")
    if (file.content_type or "application/octet-stream") not in VIDEO_TYPES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"unsupported content type {file.content_type}")
    backlog = await core.pool.fetchval("SELECT count(*) FROM job WHERE status = 'queued'")
    if backlog >= core.settings.max_job_backlog:
        metrics.REJECTED.labels("job_backlog").inc()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "job backlog full, retry later",
                            headers={"Retry-After": "30"})

    job_id = uuid.uuid4()
    key = f"jobs/{job_id}/input{_ext(file.filename)}"
    limit = core.settings.max_video_bytes
    # Spool to disk (not memory) while enforcing the size cap, then stream to storage.
    with tempfile.SpooledTemporaryFile(max_size=8 << 20) as spool:
        size = 0
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"video larger than {limit} bytes")
            await asyncio.to_thread(spool.write, chunk)
        if size == 0:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "empty upload")
        spool.seek(0)
        await core.storage.upload_fileobj(spool, key, file.content_type or "application/octet-stream")

    params = {"sample_fps": sample_fps, "conf": conf, "annotate": annotate, "track": track,
              "filename": file.filename, "bytes": size}
    await core.pool.execute(
        "INSERT INTO job (id, tenant_id, model, params, input_key) VALUES ($1, $2::uuid, $3, $4, $5)",
        job_id, p.tenant_id, model, json.dumps(params), key)
    await core.redis.xadd(JOB_STREAM, {"job_id": str(job_id)})
    return {"job_id": str(job_id), "status": "queued"}


def _ext(name: Optional[str]) -> str:
    if name and "." in name:
        ext = "." + name.rsplit(".", 1)[1].lower()
        if ext.isascii() and len(ext) <= 6 and ext[1:].isalnum():
            return ext
    return ".mp4"


async def _load(core, p: Principal, job_id: str):
    try:
        row = await core.pool.fetchrow(f"SELECT {JOB_COLUMNS} FROM job WHERE id::text = $1 AND tenant_id::text = $2",
                                       job_id, p.tenant_id)
    except Exception:
        row = None
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such job")
    return row


@router.get("/v1/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: str, request: Request, p: Principal = Depends(require("jobs"))):
    core = state(request)
    return await _job_out(core, await _load(core, p, job_id))


@router.get("/v1/jobs", response_model=list[JobOut])
async def list_jobs(request: Request, limit: int = Query(50, ge=1, le=500), p: Principal = Depends(require("jobs"))):
    core = state(request)
    rows = await core.pool.fetch(
        f"SELECT {JOB_COLUMNS} FROM job WHERE tenant_id::text = $1 ORDER BY created_at DESC LIMIT $2", p.tenant_id, limit)
    return [await _job_out(core, r) for r in rows]


@router.post("/v1/jobs/{job_id}/cancel", response_model=JobOut)
async def cancel_job(job_id: str, request: Request, p: Principal = Depends(require("jobs"))):
    core = state(request)
    row = await _load(core, p, job_id)
    await core.pool.execute(
        "UPDATE job SET status = 'cancelled', finished_at = now() WHERE id = $1 AND status IN ('queued', 'processing')",
        row["id"])
    return await _job_out(core, await _load(core, p, job_id))


@router.websocket("/v1/jobs/{job_id}/stream")
async def job_stream(ws: WebSocket, job_id: str):
    """Progress messages until the job finishes. Auth: X-API-Key header or ?api_key=."""
    p = await ws_principal(ws, "jobs")
    if p is None:
        await ws.close(code=4401)
        return
    core = ws.app.state.core
    try:
        await _load(core, p, job_id)
    except HTTPException:
        await ws.close(code=4404)
        return
    await ws.accept()
    pubsub = core.redis.pubsub()
    await pubsub.subscribe(f"job:{job_id}")
    try:
        while True:
            job = await _job_out(core, await _load(core, p, job_id))
            await ws.send_text(job.model_dump_json())
            if job.status in ("completed", "failed", "cancelled"):
                break
            await pubsub.get_message(ignore_subscribe_messages=True, timeout=2.0)
        await ws.close()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await pubsub.unsubscribe()
        await pubsub.aclose()
