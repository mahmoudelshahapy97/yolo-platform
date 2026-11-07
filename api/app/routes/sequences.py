"""Sequence inference: classify pose-keypoint windows (e.g. shoplifting_seq).

A window is one person's pose over time. Send either raw keypoints (normalised
x, y per COCO keypoint, per sample) and the server builds the model's features
exactly as cameras do, or the features themselves.
"""

import time
import uuid
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status

from yolo_platform_shared.behavior import NUM_FEATURES, NUM_FRAMES, NUM_KEYPOINTS, windows_from_sequence

from .. import metrics
from ..deps import check_model, require, state
from ..schemas import SequenceIn, SequenceResponse
from ..services.auth import Principal
from ..services.inference_service import ModelNotServing
from ..services.triton_client import TritonUnavailable

router = APIRouter(prefix="/v1/infer", tags=["inference"])

MAX_WINDOWS = 64


def _windows(body: SequenceIn) -> np.ndarray:
    if (body.keypoints is None) == (body.features is None):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "send exactly one of keypoints or features")
    items = body.keypoints if body.keypoints is not None else body.features
    if not items:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "no windows")
    if len(items) > MAX_WINDOWS:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"at most {MAX_WINDOWS} windows per request")
    if body.features is not None:
        try:
            arr = np.asarray(body.features, dtype=np.float32)
        except (ValueError, TypeError):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "features must be numbers [N][120][17][12]")
        if arr.shape[1:] != (NUM_FRAMES, NUM_KEYPOINTS, NUM_FEATURES):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"features must be [N][{NUM_FRAMES}][{NUM_KEYPOINTS}][{NUM_FEATURES}], got {list(arr.shape)}")
        return arr
    out = []
    for n, seq in enumerate(body.keypoints):
        if not isinstance(seq, list) or not 1 <= len(seq) <= NUM_FRAMES:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                f"window {n}: 1 to {NUM_FRAMES} samples of [{NUM_KEYPOINTS}][2]")
        samples: List[Optional[np.ndarray]] = []
        for t, kp in enumerate(seq):
            if kp is None:
                samples.append(None)
                continue
            try:
                a = np.asarray(kp, dtype=np.float32)
            except (ValueError, TypeError):
                a = None
            if a is None or a.ndim != 2 or a.shape[0] != NUM_KEYPOINTS or a.shape[1] not in (2, 3) \
                    or not np.isfinite(a).all():
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                    f"window {n}, sample {t}: expected [{NUM_KEYPOINTS}][2] normalised keypoints")
            samples.append(a[:, :2])
        out.append(windows_from_sequence(samples))
    return np.stack(out)


@router.post("/sequences", response_model=SequenceResponse)
async def infer_sequences(body: SequenceIn, request: Request, p: Principal = Depends(require("infer"))):
    core = state(request)
    check_model(p, body.model)
    windows = _windows(body)
    if len(windows) > 1:
        # One rate-limit token per window, like /batch per image (the dependency charged one).
        allowed, _, retry = await core.limiter.take(p.key_id, p.rate_per_min, len(windows) - 1)
        if not allowed:
            metrics.REJECTED.labels("rate_limit").inc()
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded",
                                headers={"Retry-After": str(max(1, int(retry + 0.999)))})
    started = time.perf_counter()
    try:
        model, probs, timing = await core.inference.sequences(windows, body.model)
    except ModelNotServing as exc:
        metrics.ERRORS.labels(body.model, "sequences", "not_serving").inc()
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    except TritonUnavailable as exc:
        metrics.ERRORS.labels(body.model, "sequences", "triton_unavailable").inc()
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"inference server busy: {exc}",
                            headers={"Retry-After": "1"})
    total = time.perf_counter() - started
    metrics.REQUESTS.labels(body.model, "sequences", "ok").inc()
    metrics.LATENCY.labels(body.model, "sequences").observe(total)
    names = model.names
    results = []
    for i, pr in enumerate(probs):
        top = int(np.argmax(pr))
        results.append({"index": i, "class_id": top, "class_name": names.get(top, str(top)),
                        "confidence": round(float(pr[top]), 4),
                        "probabilities": {names.get(k, str(k)): round(float(v), 4) for k, v in enumerate(pr)}})
    return SequenceResponse(request_id=f"req_{uuid.uuid4().hex[:16]}", model=model.name,
                            model_version=str(model.version), task=model.task,
                            timing=timing.to_dict(total * 1000), results=results)
