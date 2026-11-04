"""ByteTrack (Zhang et al., 2022), numpy + scipy only. One tracker per camera.

Two-stage association: confident detections first, then low-confidence ones
are used only to keep existing tracks alive (occlusion, motion blur). Matching is
IoU on Kalman-predicted boxes, fused with detection score, and boxes of different
classes never match. Tracks are not shared between cameras: two cameras may
legitimately see different objects with the same track id.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from yolo_platform_shared.decode import box_iou

TRACKED, LOST, REMOVED = 0, 1, 2


class KalmanXYAH:
    """Constant-velocity Kalman filter on (cx, cy, aspect, height) as in ByteTrack/DeepSORT."""

    _w_pos = 1.0 / 20
    _w_vel = 1.0 / 160

    def __init__(self) -> None:
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.H = np.eye(4, 8)

    def initiate(self, z: np.ndarray):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self._w_pos * h, 2 * self._w_pos * h, 1e-2, 2 * self._w_pos * h,
               10 * self._w_vel * h, 10 * self._w_vel * h, 1e-5, 10 * self._w_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self._w_pos * h, self._w_pos * h, 1e-2, self._w_pos * h,
               self._w_vel * h, self._w_vel * h, 1e-5, self._w_vel * h]
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + np.diag(np.square(std))
        return mean, cov

    def update(self, mean, cov, z):
        h = mean[3]
        std = [self._w_pos * h, self._w_pos * h, 1e-1, self._w_pos * h]
        S = self.H @ cov @ self.H.T + np.diag(np.square(std))
        K = cov @ self.H.T @ np.linalg.inv(S)
        mean = mean + K @ (z - self.H @ mean)
        cov = cov - K @ S @ K.T
        return mean, cov


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(m: np.ndarray) -> np.ndarray:
    w = m[2] * m[3]
    return np.array([m[0] - w / 2, m[1] - m[3] / 2, m[0] + w / 2, m[1] + m[3] / 2], dtype=np.float32)


@dataclass
class Track:
    track_id: int
    mean: np.ndarray
    cov: np.ndarray
    score: float
    cls: int
    state: int
    activated: bool
    start_frame: int
    end_frame: int
    hits: int = 1
    det_index: int = -1

    @property
    def box(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean)


def _match(cost: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    c = np.where(cost > thresh, thresh + 1e5, cost)
    rows, cols = linear_sum_assignment(c)
    pairs = [(r, k) for r, k in zip(rows, cols) if cost[r, k] <= thresh]
    mr = {r for r, _ in pairs}
    mc = {k for _, k in pairs}
    return pairs, [r for r in range(cost.shape[0]) if r not in mr], [k for k in range(cost.shape[1]) if k not in mc]


class ByteTracker:
    def __init__(self, frame_rate: float = 5.0, track_thresh: float = 0.45, low_thresh: float = 0.1,
                 match_thresh: float = 0.8, buffer_s: float = 6.0) -> None:
        self.kf = KalmanXYAH()
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.new_thresh = track_thresh + 0.1
        self.match_thresh = match_thresh
        self.max_lost = max(1, int(round(buffer_s * frame_rate)))
        self.frame = 0
        self.tracks: List[Track] = []
        self._next_id = 1

    def set_frame_rate(self, fps: float, buffer_s: float = 6.0) -> None:
        self.max_lost = max(1, int(round(buffer_s * fps)))

    def _cost(self, tracks: List[Track], boxes: np.ndarray, scores: np.ndarray, cls: np.ndarray, fuse: bool) -> np.ndarray:
        if not tracks or not len(boxes):
            return np.zeros((len(tracks), len(boxes)))
        iou = box_iou(np.stack([t.box for t in tracks]), boxes)
        if fuse:
            iou = iou * scores[None, :]
        cost = 1.0 - iou
        cost[np.array([t.cls for t in tracks])[:, None] != cls[None, :]] = 1.0
        return cost

    def _update(self, t: Track, box, score, cls, idx) -> None:
        t.mean, t.cov = self.kf.update(t.mean, t.cov, xyxy_to_xyah(box))
        t.score, t.cls, t.state, t.end_frame, t.det_index = float(score), int(cls), TRACKED, self.frame, idx
        t.hits += 1
        t.activated = True

    def update(self, boxes: np.ndarray, scores: np.ndarray, cls: np.ndarray) -> List[Tuple[int, int]]:
        """boxes [N,4] xyxy, scores [N], cls [N] -> [(det_index, track_id)] for tracks confirmed this frame."""
        self.frame += 1
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        cls = np.asarray(cls, dtype=np.int64).reshape(-1)
        for t in self.tracks:
            t.det_index = -1
            if t.state != REMOVED:
                if t.state != TRACKED:
                    t.mean[7] = 0.0
                t.mean, t.cov = self.kf.predict(t.mean, t.cov)

        hi = np.where(scores >= self.track_thresh)[0]
        lo = np.where((scores >= self.low_thresh) & (scores < self.track_thresh))[0]
        confirmed = [t for t in self.tracks if t.state == TRACKED and t.activated]
        unconfirmed = [t for t in self.tracks if t.state == TRACKED and not t.activated]
        pool = confirmed + [t for t in self.tracks if t.state == LOST]

        # 1) confident detections vs confirmed + lost tracks
        pairs, rem_t, rem_d = _match(self._cost(pool, boxes[hi], scores[hi], cls[hi], True), self.match_thresh)
        for ti, di in pairs:
            self._update(pool[ti], boxes[hi[di]], scores[hi[di]], cls[hi[di]], int(hi[di]))
        left_hi = hi[rem_d]

        # 2) low-confidence detections keep still-tracked tracks alive
        pool2 = [pool[i] for i in rem_t if pool[i].state == TRACKED]
        pairs, rem_t2, _ = _match(self._cost(pool2, boxes[lo], scores[lo], cls[lo], False), 0.5)
        for ti, di in pairs:
            self._update(pool2[ti], boxes[lo[di]], scores[lo[di]], cls[lo[di]], int(lo[di]))
        for i in rem_t2:
            pool2[i].state = LOST

        # 3) unconfirmed (one-frame) tracks vs remaining confident detections
        pairs, rem_u, rem_d3 = _match(self._cost(unconfirmed, boxes[left_hi], scores[left_hi], cls[left_hi], True), 0.7)
        for ti, di in pairs:
            self._update(unconfirmed[ti], boxes[left_hi[di]], scores[left_hi[di]], cls[left_hi[di]], int(left_hi[di]))
        for i in rem_u:
            unconfirmed[i].state = REMOVED

        # 4) new tracks
        for di in left_hi[rem_d3]:
            if scores[di] < self.new_thresh:
                continue
            mean, cov = self.kf.initiate(xyxy_to_xyah(boxes[di]))
            self.tracks.append(Track(self._next_id, mean, cov, float(scores[di]), int(cls[di]), TRACKED,
                                     activated=self.frame == 1, start_frame=self.frame, end_frame=self.frame,
                                     det_index=int(di)))
            self._next_id += 1

        for t in self.tracks:
            if t.state == LOST and self.frame - t.end_frame > self.max_lost:
                t.state = REMOVED
        self.tracks = [t for t in self.tracks if t.state != REMOVED]
        return [(t.det_index, t.track_id) for t in self.tracks
                if t.state == TRACKED and t.activated and t.det_index >= 0]
