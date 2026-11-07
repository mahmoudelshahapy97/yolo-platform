"""Synchronous image inference: one image or a batch; detection, pose or segmentation (+ crop
classification), or classification."""

import asyncio
import io
import time
import uuid
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from PIL import Image

from .. import metrics
from ..deps import check_model, require, state
from ..schemas import InferResponse
from ..services.auth import Principal
from ..services.inference_service import ModelNotServing
from ..services.triton_client import TritonUnavailable

router = APIRouter(prefix="/v1/infer", tags=["inference"])


def _parse_ids(raw: Optional[str], field: str) -> Optional[List[int]]:
    if raw is None or raw.strip() == "":
        return None
    try:
        return [int(x) for x in raw.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"{field} must be comma-separated class ids")


async def _read_images(request: Request, files: List[UploadFile]) -> List[np.ndarray]:
    s = state(request).settings
    if not files:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "no image uploaded")
    if len(files) > s.max_batch_images:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"at most {s.max_batch_images} images per request")
    images = []
    for f in files:
        data = await f.read(s.max_image_bytes + 1)
        if len(data) > s.max_image_bytes:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                f"{f.filename}: larger than {s.max_image_bytes} bytes")
        img = await asyncio.to_thread(_decode, data, s.max_image_pixels)
        if img is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"{f.filename}: not a decodable image")
        images.append(img)
    return images


def _decode(data: bytes, max_pixels: int) -> Optional[np.ndarray]:
    # Read only the header first, so a decompression bomb is rejected before decoding.
    try:
        with Image.open(io.BytesIO(data)) as probe:
            w, h = probe.size
    except Exception:
        return None
    if w * h > max_pixels:
        return None
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.shape[0] * img.shape[1] > max_pixels:
        return None
    return img


async def _run(request: Request, p: Principal, endpoint: str, files: List[UploadFile], model: str, conf: float,
               classes: Optional[str], max_det: int, classify_model: Optional[str], classify_classes: Optional[str],
               top_k: int) -> InferResponse:
    core = state(request)
    check_model(p, model)
    if classify_model:
        check_model(p, classify_model)
    if core.inflight >= core.settings.max_inflight:
        metrics.REJECTED.labels("inflight").inc()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "inference capacity exhausted, retry shortly",
                            headers={"Retry-After": "1"})
    started = time.perf_counter()
    core.inflight += 1
    metrics.INFLIGHT.inc()
    try:
        images = await _read_images(request, files)
        active = await core.registry.active(model)
        if active is None:
            raise ModelNotServing(f"model '{model}' has no active version")
        if active.task == "sequence":
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"'{model}' classifies pose sequences, not images: use POST /v1/infer/sequences")
        if active.task == "classify":
            m, results, timing = await core.inference.classify(images, model, top_k=top_k)
            cm = None
        else:
            m, cm, results, timing = await core.inference.detect(
                images, model, conf=conf, classes=_parse_ids(classes, "classes"), max_det=max_det,
                classify_model=classify_model, classify_classes=_parse_ids(classify_classes, "classify_classes"),
                top_k=top_k)
    except ModelNotServing as exc:
        metrics.ERRORS.labels(model, endpoint, "not_serving").inc()
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    except TritonUnavailable as exc:
        metrics.ERRORS.labels(model, endpoint, "triton_unavailable").inc()
        metrics.REJECTED.labels("triton_queue").inc()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"inference server busy: {exc}",
                            headers={"Retry-After": "1"})
    except HTTPException as exc:
        metrics.ERRORS.labels(model, endpoint, f"http_{exc.status_code}").inc()
        raise
    finally:
        core.inflight -= 1
        metrics.INFLIGHT.dec()

    total = time.perf_counter() - started
    metrics.REQUESTS.labels(model, endpoint, "ok").inc()
    metrics.LATENCY.labels(model, endpoint).observe(total)
    metrics.IMAGES.labels(model).inc(len(images))
    for stage_name in ("preprocess", "inference", "postprocess"):
        metrics.STAGE.labels(model, stage_name).observe(getattr(timing, f"{stage_name}_ms") / 1000)
    return InferResponse(
        request_id=f"req_{uuid.uuid4().hex[:16]}", model=m.name, model_version=str(m.version), task=m.task,
        classify_model=cm.name if cm else None, classify_model_version=str(cm.version) if cm else None,
        timing=timing.to_dict(total * 1000),
        images=[{"index": i, "filename": f.filename, "width": r.width, "height": r.height,
                 "detections": r.detections, "classifications": r.classifications}
                for i, (f, r) in enumerate(zip(files, results))],
    )


@router.post("/images", response_model=InferResponse, response_model_exclude_none=True)
async def infer_image(
    request: Request,
    file: UploadFile = File(..., description="JPEG/PNG/WebP image"),
    model: str = Form("yolo26n_detection"),
    conf: float = Form(0.25, ge=0.01, le=1.0),
    classes: Optional[str] = Form(None, description="comma-separated class ids to keep"),
    max_det: int = Form(300, ge=1, le=300),
    classify_model: Optional[str] = Form(None, description="classify each detection crop with this model"),
    classify_classes: Optional[str] = Form(None, description="only classify detections of these class ids"),
    top_k: int = Form(1, ge=1, le=10),
    p: Principal = Depends(require("infer")),
):
    return await _run(request, p, "images", [file], model, conf, classes, max_det, classify_model, classify_classes, top_k)


@router.post("/batch", response_model=InferResponse, response_model_exclude_none=True)
async def infer_batch(
    request: Request,
    files: List[UploadFile] = File(..., description="up to MAX_BATCH_IMAGES images"),
    model: str = Form("yolo26n_detection"),
    conf: float = Form(0.25, ge=0.01, le=1.0),
    classes: Optional[str] = Form(None),
    max_det: int = Form(300, ge=1, le=300),
    classify_model: Optional[str] = Form(None),
    classify_classes: Optional[str] = Form(None),
    top_k: int = Form(1, ge=1, le=10),
    p: Principal = Depends(require("infer")),
):
    # The rate limit charges one token per image (checked again here: the dependency charged one).
    allowed, _, retry = await state(request).limiter.take(p.key_id, p.rate_per_min, max(0, len(files) - 1))
    if not allowed:
        metrics.REJECTED.labels("rate_limit").inc()
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded",
                            headers={"Retry-After": str(max(1, int(retry + 0.999)))})
    return await _run(request, p, "batch", files, model, conf, classes, max_det, classify_model, classify_classes, top_k)
