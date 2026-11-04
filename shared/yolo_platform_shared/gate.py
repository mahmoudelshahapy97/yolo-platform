"""Engine-vs-reference comparison, used by the parity tool and the rollout gate.

Golden files (``golden.npz`` in a model version directory) are written by
inference/validation/parity.py. They hold letterboxed UINT8 inputs and the
original .pt model's outputs on them, so the gate needs only numpy and Triton:

  detect:   inputs [N,S,S,3] uint8, ref_boxes [K,4] (model space xyxy), ref_conf [K],
            ref_cls [K], ref_img [K] (which input each box belongs to)
  classify: inputs [N,S,S,3] uint8, ref_top1 [N], ref_top1_prob [N]
"""

from typing import Dict, List, Tuple

import numpy as np

from .decode import box_iou

THRESHOLDS = {
    "detect": {"recall": 0.95, "precision": 0.95, "mean_iou": 0.90, "mean_conf_diff": 0.05},
    "classify": {"top1_agreement": 0.95, "mean_conf_diff": 0.05},
}
CONFIDENT = 0.4   # only boxes clearly above the 0.25 threshold count, so borderline flips do not fail a gate
DETECT_CONF = 0.25


def match(ref: np.ndarray, ref_cls: np.ndarray, got: np.ndarray, got_cls: np.ndarray) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one match per class at IoU >= 0.5 -> [(ref_i, got_j, iou)]."""
    if len(ref) == 0 or len(got) == 0:
        return []
    iou = box_iou(ref, got)
    iou[ref_cls[:, None] != got_cls[None, :]] = 0
    pairs, used_ref, used_got = [], set(), set()
    for flat in np.argsort(-iou, axis=None):
        i, j = divmod(int(flat), iou.shape[1])
        if iou[i, j] < 0.5:
            break
        if i in used_ref or j in used_got:
            continue
        used_ref.add(i)
        used_got.add(j)
        pairs.append((i, j, float(iou[i, j])))
    return pairs


class DetectionTally:
    """Accumulates per-image comparisons into gate metrics."""

    def __init__(self) -> None:
        self.ref_confident = self.ref_matched = self.got_confident = self.got_matched = 0
        self.ious: List[float] = []
        self.conf_diffs: List[float] = []
        self.images = 0

    def add(self, ref_boxes, ref_conf, ref_cls, got_boxes, got_conf, got_cls) -> None:
        self.images += 1
        pairs = match(ref_boxes, ref_cls, got_boxes, got_cls)
        matched_ref = {i for i, _, _ in pairs}
        matched_got = {j for _, j, _ in pairs}
        self.ref_confident += int((ref_conf >= CONFIDENT).sum())
        self.ref_matched += sum(1 for i in matched_ref if ref_conf[i] >= CONFIDENT)
        self.got_confident += int((got_conf >= CONFIDENT).sum())
        self.got_matched += sum(1 for j in matched_got if got_conf[j] >= CONFIDENT)
        for i, j, iou in pairs:
            self.ious.append(iou)
            self.conf_diffs.append(abs(float(ref_conf[i]) - float(got_conf[j])))

    def result(self) -> Dict:
        t = THRESHOLDS["detect"]
        r = {
            "images": self.images,
            "ref_confident_boxes": self.ref_confident,
            "recall": round(self.ref_matched / max(self.ref_confident, 1), 4),
            "precision": round(self.got_matched / max(self.got_confident, 1), 4),
            "mean_iou": round(float(np.mean(self.ious)), 4) if self.ious else 0.0,
            "mean_conf_diff": round(float(np.mean(self.conf_diffs)), 4) if self.conf_diffs else 1.0,
        }
        r["passed"] = bool(
            self.ref_confident > 0 and r["recall"] >= t["recall"] and r["precision"] >= t["precision"]
            and r["mean_iou"] >= t["mean_iou"] and r["mean_conf_diff"] <= t["mean_conf_diff"]
        )
        r["thresholds"] = t
        return r


def engine_rows(out: np.ndarray, conf: float = DETECT_CONF):
    """One image's end-to-end output [max_det,6] -> (boxes, conf, cls), model space."""
    rows = out[out[:, 4] >= conf]
    return rows[:, :4], rows[:, 4], rows[:, 5].astype(int)


def evaluate_detect(golden: Dict[str, np.ndarray], outputs: np.ndarray) -> Dict:
    tally = DetectionTally()
    for n, out in enumerate(outputs):
        sel = golden["ref_img"] == n
        got_boxes, got_conf, got_cls = engine_rows(out)
        tally.add(golden["ref_boxes"][sel], golden["ref_conf"][sel], golden["ref_cls"][sel].astype(int),
                  got_boxes, got_conf, got_cls)
    return tally.result()


def evaluate_classify(golden: Dict[str, np.ndarray], outputs: np.ndarray) -> Dict:
    top1 = outputs.argmax(axis=1)
    agree = float((top1 == golden["ref_top1"]).mean()) if len(top1) else 0.0
    diff = float(np.abs(outputs[np.arange(len(outputs)), golden["ref_top1"]] - golden["ref_top1_prob"]).mean()) if len(top1) else 1.0
    t = THRESHOLDS["classify"]
    return {
        "images": int(len(outputs)),
        "top1_agreement": round(agree, 4),
        "mean_conf_diff": round(diff, 4),
        "passed": bool(len(outputs) > 0 and agree >= t["top1_agreement"] and diff <= t["mean_conf_diff"]),
        "thresholds": t,
    }
