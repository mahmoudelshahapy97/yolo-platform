"""Letterbox and crop maths, matching Ultralytics exactly.

The engines take UINT8 NHWC RGB (see inference/export/wrap_uint8.py), so
preprocessing is only geometry: resize, pad with 114 and channel order. The
normalisation happens on the GPU inside the engine.

Ultralytics' LetterBox (auto=False, center=True, scaleup=True) is reproduced
including its rounding, so boxes map back to the same pixels the .pt gives.
"""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

PAD_VALUE = 114


@dataclass(frozen=True)
class LetterboxMeta:
    """What is needed to map model coordinates back to the original image."""

    orig_w: int
    orig_h: int
    ratio: float
    pad_left: int
    pad_top: int


def letterbox_params(orig_w: int, orig_h: int, size: int) -> Tuple[int, int, int, int, int, int, float]:
    """(new_w, new_h, top, bottom, left, right, ratio) as Ultralytics computes them."""
    r = min(size / orig_h, size / orig_w)
    new_w, new_h = int(round(orig_w * r)), int(round(orig_h * r))
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return new_w, new_h, top, bottom, left, right, r


def letterbox(image_bgr: np.ndarray, size: int, out: np.ndarray | None = None) -> Tuple[np.ndarray, LetterboxMeta]:
    """BGR HxWx3 uint8 -> RGB size x size x 3 uint8, plus the mapping metadata.

    ``out`` may be a preallocated (size, size, 3) uint8 view (for example a slot in
    a shared-memory batch buffer) to avoid a copy.
    """
    h, w = image_bgr.shape[:2]
    new_w, new_h, top, bottom, left, right, r = letterbox_params(w, h, size)
    resized = image_bgr if (w, h) == (new_w, new_h) else cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if out is None:
        out = np.empty((size, size, 3), dtype=np.uint8)
    out[...] = PAD_VALUE
    # BGR -> RGB while copying into the padded canvas.
    out[top:top + new_h, left:left + new_w] = resized[..., ::-1]
    return out, LetterboxMeta(orig_w=w, orig_h=h, ratio=r, pad_left=left, pad_top=top)


def scale_boxes(boxes_xyxy: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Model-space xyxy -> original-image xyxy, clipped to the image (Ultralytics scale_boxes)."""
    b = boxes_xyxy.astype(np.float32, copy=True)
    b[:, [0, 2]] -= meta.pad_left
    b[:, [1, 3]] -= meta.pad_top
    b /= meta.ratio
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, meta.orig_w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, meta.orig_h)
    return b


def classify_crop(image_bgr: np.ndarray, size: int, out: np.ndarray | None = None) -> np.ndarray:
    """Ultralytics classify_transforms: resize the short side to ``size``, centre-crop, RGB."""
    h, w = image_bgr.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("empty crop")
    scale = size / min(h, w)
    new_w, new_h = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)
    top, left = (new_h - size) // 2, (new_w - size) // 2
    if out is None:
        out = np.empty((size, size, 3), dtype=np.uint8)
    out[...] = resized[top:top + size, left:left + size, ::-1]
    return out
