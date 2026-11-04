"""Turn raw engine outputs into plain detection / classification records.

Every detector in the repository has an end-to-end output [N, max_det, 6]
(x1, y1, x2, y2, confidence, class) in letterboxed model space: YOLO26 heads
natively, older heads with NMS embedded at export. Rows are sorted by
confidence; padding rows have confidence 0.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

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

    def to_dict(self) -> dict:
        d = asdict(self)
        bbox = {k: round(d.pop(k), 2) for k in ("x1", "y1", "x2", "y2")}
        d["confidence"] = round(d["confidence"], 4)
        d["bbox"] = bbox
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
    """One image's [max_det, 6] output -> Detections in original-image pixels."""
    rows = out[out[:, 4] >= conf]
    if classes is not None:
        rows = rows[np.isin(rows[:, 5].astype(int), np.asarray(list(classes), dtype=int))]
    rows = rows[np.argsort(-rows[:, 4], kind="stable")][:max_det]
    if not len(rows):
        return []
    boxes = scale_boxes(rows[:, :4], meta)
    result = []
    for (x1, y1, x2, y2), score, cls in zip(boxes, rows[:, 4], rows[:, 5]):
        cid = int(cls)
        result.append(Detection(cid, names.get(cid, str(cid)), float(score), float(x1), float(y1), float(x2), float(y2)))
    return result


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
