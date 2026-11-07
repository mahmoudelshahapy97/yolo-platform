"""Turn raw engine outputs into plain detection / classification records.

Every box model in the repository has an end-to-end output [N, max_det, 6 + extras]
(x1, y1, x2, y2, confidence, class, ...) in letterboxed model space, NMS embedded
at export. Rows are sorted by confidence; padding rows have confidence 0.

  detect    no extras
  pose      K keypoints x (x, y, conf), model space
  segment   32 mask coefficients; a second output holds the prototypes [32, S/4, S/4]
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .preprocess import LetterboxMeta, scale_boxes


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    # pose: [K, 3] (x, y, conf) in original-image pixels
    keypoints: Optional[np.ndarray] = None
    # segment: outline of the largest mask region, [P, 2] original-image pixels
    polygon: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        d = {"class_id": self.class_id, "class_name": self.class_name, "confidence": round(self.confidence, 4),
             "bbox": {k: round(getattr(self, k), 2) for k in ("x1", "y1", "x2", "y2")}}
        if self.keypoints is not None:
            d["keypoints"] = [[round(float(x), 2), round(float(y), 2), round(float(c), 4)] for x, y, c in self.keypoints]
        if self.polygon is not None:
            d["polygon"] = [[round(float(x), 1), round(float(y), 1)] for x, y in self.polygon]
        return d

    @property
    def xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


@dataclass
class Classification:
    class_id: int
    class_name: str
    confidence: float

    def to_dict(self) -> dict:
        return {"class_id": self.class_id, "class_name": self.class_name, "confidence": round(self.confidence, 4)}


def decode_detections(
    out: np.ndarray,
    meta: LetterboxMeta,
    names: Dict[int, str],
    conf: float = 0.25,
    classes: Optional[Sequence[int]] = None,
    max_det: int = 300,
) -> List[Detection]:
    """One image's [max_det, 6+] output -> Detections in original-image pixels (extras ignored)."""
    rows = select_rows(out, conf, classes, max_det)
    return _boxes(rows, meta, names)


# The FP16 engines' embedded NMS occasionally keeps a near-identical second box of the
# same class (IoU 0.93-0.97 in parity, never in the FP32 graph). Rows are suppressed again
# here at Ultralytics' default IoU, so clients and the rollout gate see one box per object.
DEDUPE_IOU = 0.7


def dedupe(rows: np.ndarray, iou: float = DEDUPE_IOU) -> np.ndarray:
    """Greedy class-aware NMS over rows already sorted best first."""
    if len(rows) < 2:
        return rows
    ious = box_iou(rows[:, :4], rows[:, :4])
    same = rows[:, 5][:, None] == rows[:, 5][None, :]
    suppress = (ious > iou) & same
    keep = np.ones(len(rows), dtype=bool)
    for i in range(len(rows)):
        if keep[i]:
            later = suppress[i].copy()
            later[: i + 1] = False
            keep &= ~later
    return rows[keep]


def select_rows(out: np.ndarray, conf: float, classes: Optional[Sequence[int]], max_det: int) -> np.ndarray:
    """Rows above ``conf`` (of ``classes``), best first, duplicates removed, at most ``max_det``."""
    rows = out[out[:, 4] >= conf]
    if classes is not None:
        rows = rows[np.isin(rows[:, 5].astype(int), np.asarray(list(classes), dtype=int))]
    return dedupe(rows[np.argsort(-rows[:, 4], kind="stable")])[:max_det]


def _boxes(rows: np.ndarray, meta: LetterboxMeta, names: Dict[int, str]) -> List[Detection]:
    if not len(rows):
        return []
    boxes = scale_boxes(rows[:, :4], meta)
    result = []
    for (x1, y1, x2, y2), score, cls in zip(boxes, rows[:, 4], rows[:, 5]):
        cid = int(cls)
        result.append(Detection(cid, names.get(cid, str(cid)), float(score), float(x1), float(y1), float(x2), float(y2)))
    return result


def scale_points(xy: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Model-space points [..., 2] -> original-image pixels, clipped (Ultralytics scale_coords)."""
    p = xy.astype(np.float32, copy=True)
    p[..., 0] = ((p[..., 0] - meta.pad_left) / meta.ratio).clip(0, meta.orig_w)
    p[..., 1] = ((p[..., 1] - meta.pad_top) / meta.ratio).clip(0, meta.orig_h)
    return p


def decode_pose(
    out: np.ndarray,
    meta: LetterboxMeta,
    names: Dict[int, str],
    kpt_shape: Tuple[int, int] = (17, 3),
    conf: float = 0.25,
    classes: Optional[Sequence[int]] = None,
    max_det: int = 300,
) -> List[Detection]:
    """One image's [max_det, 6 + K*3] pose output -> Detections with keypoints in original pixels."""
    rows = select_rows(out, conf, classes, max_det)
    dets = _boxes(rows, meta, names)
    k, dims = kpt_shape
    if dets:
        kpts = rows[:, 6:6 + k * dims].reshape(len(rows), k, dims).astype(np.float32)
        kpts[..., :2] = scale_points(kpts[..., :2], meta)
        if dims == 2:  # no visibility column: treat every point as visible
            kpts = np.concatenate([kpts, np.ones((len(rows), k, 1), np.float32)], axis=2)
        for d, kp in zip(dets, kpts):
            d.keypoints = kp
    return dets


def mask_logits(coeffs: np.ndarray, protos: np.ndarray) -> np.ndarray:
    """coefficients [n, 32] @ prototypes [32, mh, mw] -> mask logits [n, mh, mw] (sigmoid > 0.5 == logit > 0)."""
    nm, mh, mw = protos.shape
    if not len(coeffs):
        return np.zeros((0, mh, mw), np.float32)
    return (coeffs.astype(np.float32) @ protos.reshape(nm, -1).astype(np.float32)).reshape(-1, mh, mw)


def full_masks(coeffs: np.ndarray, protos: np.ndarray, boxes: np.ndarray, imgsz: int) -> np.ndarray:
    """Binary masks over the whole model input [n, S, S], as Ultralytics 8.4 process_mask(upsample=True):
    logits upsampled bilinearly to the input size, cropped to the box at that resolution, > 0."""
    logits = mask_logits(coeffs, protos)
    out = np.zeros((len(logits), imgsz, imgsz), dtype=bool)
    grid = np.arange(imgsz, dtype=np.float32)
    for k, (lg, b) in enumerate(zip(logits, boxes.astype(np.float32))):
        up = cv2.resize(lg, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        cols = (grid >= b[0]) & (grid < b[2])
        rows = (grid >= b[1]) & (grid < b[3])
        out[k] = (up > 0) & rows[:, None] & cols[None, :]
    return out


def decode_segment(
    out: np.ndarray,
    protos: np.ndarray,
    meta: LetterboxMeta,
    names: Dict[int, str],
    imgsz: int,
    conf: float = 0.25,
    classes: Optional[Sequence[int]] = None,
    max_det: int = 300,
) -> List[Detection]:
    """One image's [max_det, 6 + 32] output plus its [32, S/4, S/4] prototypes -> Detections with polygons.

    The same mask as ``full_masks`` (Ultralytics), but each box's logits are
    upsampled only over the box itself. The polygon is the largest outer contour,
    mapped to original pixels.
    """
    rows = select_rows(out, conf, classes, max_det)
    dets = _boxes(rows, meta, names)
    if not dets:
        return dets
    nm, mh, mw = protos.shape
    logits = mask_logits(rows[:, 6:6 + nm], protos)
    scale = imgsz / mw
    vis_x0, vis_y0 = meta.pad_left, meta.pad_top
    vis_x1 = vis_x0 + int(round(meta.orig_w * meta.ratio))
    vis_y1 = vis_y0 + int(round(meta.orig_h * meta.ratio))
    for d, lg, box in zip(dets, logits, rows[:, :4].astype(np.float32)):
        # Pixels c with box_x1 <= c < box_x2 (crop_mask), within the visible (unpadded) area.
        x0, y0 = max(int(np.ceil(box[0])), vis_x0), max(int(np.ceil(box[1])), vis_y0)
        x1, y1 = min(int(np.ceil(box[2])), vis_x1), min(int(np.ceil(box[3])), vis_y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            d.polygon = np.zeros((0, 2), np.float32)
            continue
        # Bilinear upsampling (half-pixel centres, edge clamp, like cv2.resize) of this region only.
        gx = (np.arange(x0, x1, dtype=np.float32) + 0.5) / scale - 0.5
        gy = (np.arange(y0, y1, dtype=np.float32) + 0.5) / scale - 0.5
        mapx, mapy = np.meshgrid(gx, gy)
        roi = cv2.remap(lg, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = (roi > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            d.polygon = np.zeros((0, 2), np.float32)
            continue
        c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        c[:, 0] += x0
        c[:, 1] += y0
        d.polygon = scale_points(c, meta)
    return dets


def decode_classification(probs: np.ndarray, names: Dict[int, str], top_k: int = 5) -> List[Classification]:
    """One image's softmax vector -> top-k Classifications."""
    top_k = max(1, min(top_k, probs.shape[-1]))
    idx = np.argpartition(-probs, top_k - 1)[:top_k]
    idx = idx[np.argsort(-probs[idx], kind="stable")]
    return [Classification(int(i), names.get(int(i), str(int(i))), float(probs[i])) for i in idx]


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy arrays a [N,4] and b [M,4]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)
