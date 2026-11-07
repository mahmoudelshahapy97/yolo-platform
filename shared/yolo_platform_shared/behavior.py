"""Per-person shoplifting detection over a sliding window of pose features (numpy only).

The features and the windowing are ported from InsightEye's shoplifting model as
the standalone shoplifting app runs it (backend/app/services/shoplifting_model.py
and behavior_service.py there). The network itself is served by Triton
(``shoplifting_seq``, which z-scores each window and returns softmax
probabilities), so this module never imports torch.

Every tracked person has their own window:

    sample   at ``sample_fps``, append extract_features(person's keypoints);
             while the tracker has lost them, an all-zero row
    judge    once ``min_window_samples`` of them are real, on every new sample
             (sliding window): the Shoplifting probability, averaged over the last
             ``smooth_window`` judgements. A window not yet full is left-padded
             with zero rows.
    alert    when the smoothed probability reaches ``threshold``, at most once
             per person per ``track_cooldown_s`` and per camera per ``camera_cooldown_s``

A new track id that appears within ``handover_s`` of a person vanishing, with a
similar body height, takes over that person's window (ByteTrack re-ids people
after occlusions).

Judging is split from sampling so the classifier can run elsewhere (the camera
shard's behaviour thread): ``update()`` returns the windows due for a judgement,
``judge()`` takes each probability back. Not thread-safe: callers hold a lock.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Fixed by training.
NUM_FRAMES = 120       # T: samples per window
NUM_KEYPOINTS = 17     # V: COCO keypoints
NUM_FEATURES = 12      # C: features per keypoint (7 used, zero-padded)
SHOPLIFTING_CLASS = 1  # "Shoplifting" in the model's output

_L_WRIST, _R_WRIST, _L_HIP, _R_HIP = 9, 10, 11, 12

# A new id whose box height is within this factor of a vanished person's can take over their window.
HANDOVER_HEIGHT_RATIO = 1.5

Box = Tuple[float, float, float, float]


def keypoints_xyn(keypoints_px: np.ndarray, width: int, height: int) -> np.ndarray:
    """[17, 2+] keypoints in original pixels -> [17, 2] normalised to the frame (Ultralytics ``xyn``)."""
    kp = np.asarray(keypoints_px, dtype=np.float32)[:, :2].copy()
    kp[:, 0] /= max(width, 1)
    kp[:, 1] /= max(height, 1)
    return kp


def extract_features(kpts: np.ndarray, prev_kpts: Optional[np.ndarray]) -> np.ndarray:
    """(17, 12) features for one person from normalised (17, 2) keypoints.

    Columns: x, y, velocity since the previous sample (dx, dy), then the
    left-wrist->mid-hip, right-wrist->mid-hip and wrist->wrist distances repeated
    on every row, zero-padded to 12 - InsightEye's layout exactly.
    """
    kpts = np.asarray(kpts, dtype=np.float32)
    vel = kpts - prev_kpts if prev_kpts is not None else np.zeros_like(kpts)
    l_hand, r_hand = kpts[_L_WRIST], kpts[_R_WRIST]
    mid_hip = (kpts[_L_HIP] + kpts[_R_HIP]) / 2.0
    dist_l = np.linalg.norm(l_hand - mid_hip)
    dist_r = np.linalg.norm(r_hand - mid_hip)
    dist_hh = np.linalg.norm(l_hand - r_hand)
    scalars = np.tile([dist_l, dist_r, dist_hh], (NUM_KEYPOINTS, 1))
    feat = np.concatenate([kpts, vel, scalars], axis=1)                       # (17, 7)
    pad = np.zeros((NUM_KEYPOINTS, NUM_FEATURES - feat.shape[1]), dtype=np.float32)
    return np.concatenate([feat, pad], axis=1).astype(np.float32)            # (17, 12)


def windows_from_sequence(xyn_seq: Sequence[Optional[np.ndarray]]) -> np.ndarray:
    """One person's keypoints over time ([T<=120] of (17, 2) normalised, None = not seen) -> (120, 17, 12).

    The same rows the live tracker builds: velocity only between consecutive
    sightings, an absent sample is a zero row, a short window is left-padded.
    """
    rows, prev = [], None
    for kp in list(xyn_seq)[-NUM_FRAMES:]:
        if kp is None:
            rows.append(np.zeros((NUM_KEYPOINTS, NUM_FEATURES), np.float32))
            prev = None
            continue
        kp = np.asarray(kp, dtype=np.float32)[:, :2]
        rows.append(extract_features(kp, prev))
        prev = kp
    return pad_window(rows)


def pad_window(rows: List[np.ndarray]) -> np.ndarray:
    if len(rows) < NUM_FRAMES:
        pad = np.zeros((NUM_FRAMES - len(rows), NUM_KEYPOINTS, NUM_FEATURES), dtype=np.float32)
        return np.concatenate([pad, np.stack(rows)]) if rows else pad
    return np.stack(rows[-NUM_FRAMES:])


@dataclass
class ShopliftingParams:
    sample_fps: float = 3.0
    threshold: float = 0.5
    smooth_window: int = 3
    min_window_samples: int = 30
    min_person_height_px: float = 60.0
    track_cooldown_s: float = 300.0
    camera_cooldown_s: float = 30.0
    track_max_misses: int = 15
    handover_s: float = 5.0


@dataclass
class Person:
    track_id: int
    box: Box                 # xyxy, original pixels
    keypoints_xyn: np.ndarray  # (17, 2) normalised
    confidence: float = 1.0

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class ShopliftingAlert:
    track_id: int
    probability: float      # smoothed
    peak: float
    box: Box
    first_seen: float
    ts: float
    alerts: int             # this person's alert count, this one included
    real_samples: int


@dataclass
class _Track:
    track_id: int
    first_seen: float
    last_seen: float
    box: Box
    window: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=NUM_FRAMES))
    real: Deque[bool] = field(default_factory=lambda: deque(maxlen=NUM_FRAMES))
    judgements: Deque[float] = field(default_factory=deque)
    prev_kpts: Optional[np.ndarray] = None
    last_sample_at: Optional[float] = None
    last_alert_at: Optional[float] = None
    alerts: int = 0
    probability: Optional[float] = None
    peak: Optional[float] = None
    misses: int = 0

    @property
    def real_samples(self) -> int:
        return sum(self.real)


class ShopliftingTracker:
    def __init__(self, params: Optional[ShopliftingParams] = None) -> None:
        self.params = params or ShopliftingParams()
        self.tracks: Dict[int, _Track] = {}
        self._camera_last_alert: Optional[float] = None

    def set_params(self, params: ShopliftingParams) -> None:
        self.params = params
        for t in self.tracks.values():
            t.judgements = deque(t.judgements, maxlen=max(1, params.smooth_window))

    # ---------------------------------------------------------------- input
    def update(self, people: List[Person], ts: float) -> List[Tuple[int, np.ndarray]]:
        """Record one processed frame's people; returns (track_id, window) pairs due for a judgement."""
        seen = {p.track_id for p in people}
        due: List[Tuple[int, np.ndarray]] = []
        for person in people:
            track = self.tracks.get(person.track_id)
            if track is None:
                track = self._handover(person, ts, seen) or self._open(person, ts)
            track.last_seen = ts
            track.box = person.box
            track.misses = 0
            if person.height < self.params.min_person_height_px:
                continue
            if self._sample(track, person, ts) and track.real_samples >= self.params.min_window_samples:
                due.append((track.track_id, pad_window(list(track.window))))
        for track in list(self.tracks.values()):
            if track.track_id in seen:
                continue
            track.misses += 1
            if track.misses > self.params.track_max_misses:
                del self.tracks[track.track_id]
            elif self._due(track, ts):
                track.window.append(np.zeros((NUM_KEYPOINTS, NUM_FEATURES), dtype=np.float32))
                track.real.append(False)
                track.prev_kpts = None
                track.last_sample_at = ts
        return due

    def judge(self, track_id: int, probability: float, ts: float) -> Optional[ShopliftingAlert]:
        """Take the classifier's Shoplifting probability for a window ``update()`` returned."""
        track = self.tracks.get(track_id)
        if track is None:
            return None  # the person left (or was handed over) while the window was being judged
        if track.judgements.maxlen != max(1, self.params.smooth_window):
            track.judgements = deque(track.judgements, maxlen=max(1, self.params.smooth_window))
        track.judgements.append(float(probability))
        smoothed = sum(track.judgements) / len(track.judgements)
        track.probability = smoothed
        track.peak = smoothed if track.peak is None else max(track.peak, smoothed)
        if smoothed < self.params.threshold:
            return None
        if track.last_alert_at is not None and ts - track.last_alert_at < self.params.track_cooldown_s:
            return None
        if self._camera_last_alert is not None and ts - self._camera_last_alert < self.params.camera_cooldown_s:
            return None
        track.last_alert_at = ts
        track.alerts += 1
        self._camera_last_alert = ts
        return ShopliftingAlert(track.track_id, smoothed, track.peak, track.box, track.first_seen, ts, track.alerts,
                                track.real_samples)

    # ---------------------------------------------------------- per track
    def _open(self, person: Person, ts: float) -> _Track:
        track = _Track(person.track_id, ts, ts, person.box,
                       judgements=deque(maxlen=max(1, self.params.smooth_window)))
        self.tracks[person.track_id] = track
        return track

    def _handover(self, person: Person, ts: float, in_view: set) -> Optional[_Track]:
        """The vanished person a new track id most likely is, re-keyed to it; or None."""
        if self.params.handover_s <= 0 or person.height <= 0:
            return None
        candidates = [
            t for t in self.tracks.values()
            if t.track_id not in in_view
            and 0 < ts - t.last_seen <= self.params.handover_s
            and 1 / HANDOVER_HEIGHT_RATIO <= (t.box[3] - t.box[1]) / person.height <= HANDOVER_HEIGHT_RATIO
        ]
        if not candidates:
            return None
        track = max(candidates, key=lambda t: (len(t.window), t.last_seen))
        del self.tracks[track.track_id]
        track.track_id = person.track_id
        track.prev_kpts = None   # a jump to a new place is not movement
        self.tracks[person.track_id] = track
        return track

    def _due(self, track: _Track, ts: float) -> bool:
        interval = 1.0 / max(0.1, self.params.sample_fps)
        # A quarter-interval of slack absorbs jitter, so processing at exactly
        # sample_fps samples every frame instead of every other one.
        return track.last_sample_at is None or ts - track.last_sample_at >= interval * 0.75

    def _sample(self, track: _Track, person: Person, ts: float) -> bool:
        if not self._due(track, ts):
            return False
        kp = np.asarray(person.keypoints_xyn, dtype=np.float32)[:, :2]
        track.window.append(extract_features(kp, track.prev_kpts))
        track.real.append(True)
        track.prev_kpts = kp.copy()
        track.last_sample_at = ts
        return True
