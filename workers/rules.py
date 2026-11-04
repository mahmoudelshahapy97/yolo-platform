"""Per-camera rules over tracked detections -> events, with cooldown deduplication.

Pure Python + numpy: no I/O, so it is tested with synthetic tracks
(tests/test_rules.py). Coordinates arrive in original-frame pixels and are
normalised by the frame size, because rule geometry is normalised.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from yolo_platform_shared.camera_config import CameraConfig, Rule


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    confidence: float
    box: Tuple[float, float, float, float]   # xyxy, pixels


@dataclass
class Event:
    rule: str
    type: str
    ts: float
    track_id: Optional[int]
    class_name: Optional[str]
    confidence: Optional[float]
    box: Optional[Tuple[float, float, float, float]]
    data: dict = field(default_factory=dict)


def point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def side_of_line(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_intersect(p1, p2, a, b) -> bool:
    d1, d2 = side_of_line(p1, a, b), side_of_line(p2, a, b)
    d3, d4 = side_of_line(a, p1, p2), side_of_line(b, p1, p2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


class RuleState:
    def __init__(self, rule: Rule) -> None:
        self.rule = rule
        self.classes = set(rule.classes)
        self.last_fired: Dict[object, float] = {}        # dedup key -> ts
        self.inside_since: Dict[int, float] = {}         # zone_dwell: track -> entered at
        self.last_point: Dict[int, Tuple[float, float]] = {}  # line_cross: track -> last anchor
        self.seen: set = set()                           # presence: tracks already reported

    def cooled(self, key, ts: float) -> bool:
        last = self.last_fired.get(key)
        return last is None or ts - last >= self.rule.cooldown_s

    def fire(self, key, ts: float) -> None:
        self.last_fired[key] = ts


class RulesEngine:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.states = [RuleState(r) for r in config.rules]

    def evaluate(self, objects: List[TrackedObject], frame_w: int, frame_h: int, ts: float) -> List[Event]:
        events: List[Event] = []
        live_ids = {o.track_id for o in objects}
        for st in self.states:
            r = st.rule
            cands = [o for o in objects if o.class_name in st.classes and o.confidence >= r.min_confidence]

            def norm_anchor(o: TrackedObject) -> Tuple[float, float]:
                x1, y1, x2, y2 = o.box
                return ((x1 + x2) / 2 / frame_w, y2 / frame_h)   # bottom centre: where the object stands

            if r.zone:
                in_zone = [o for o in cands if point_in_polygon(*norm_anchor(o), r.zone)]
            else:
                in_zone = cands

            if r.type == "presence":
                for o in in_zone:
                    if o.track_id in st.seen:
                        continue
                    st.seen.add(o.track_id)
                    if st.cooled(("class", o.class_name), ts):
                        st.fire(("class", o.class_name), ts)
                        events.append(Event(r.name, "presence", ts, o.track_id, o.class_name, o.confidence, o.box))

            elif r.type == "zone_dwell":
                now_inside = {o.track_id for o in in_zone}
                for tid in list(st.inside_since):
                    if tid not in now_inside:
                        del st.inside_since[tid]
                for o in in_zone:
                    since = st.inside_since.setdefault(o.track_id, ts)
                    dwell = ts - since
                    if dwell >= r.dwell_s and st.cooled(o.track_id, ts):
                        st.fire(o.track_id, ts)
                        events.append(Event(r.name, "zone_dwell", ts, o.track_id, o.class_name, o.confidence, o.box,
                                            {"dwell_s": round(dwell, 2)}))

            elif r.type == "line_cross":
                a, b = r.line
                for o in cands:
                    p = norm_anchor(o)
                    prev = st.last_point.get(o.track_id)
                    st.last_point[o.track_id] = p
                    if prev is None or not segments_intersect(prev, p, a, b):
                        continue
                    direction = "left_to_right" if side_of_line(prev, a, b) > 0 else "right_to_left"
                    if st.cooled((o.track_id, direction), ts):
                        st.fire((o.track_id, direction), ts)
                        events.append(Event(r.name, "line_cross", ts, o.track_id, o.class_name, o.confidence, o.box,
                                            {"direction": direction}))

            elif r.type == "count_above":
                n = len(in_zone)
                if n > r.count and st.cooled("count", ts):
                    st.fire("count", ts)
                    best = max(in_zone, key=lambda o: o.confidence)
                    events.append(Event(r.name, "count_above", ts, None, best.class_name, best.confidence, None,
                                        {"count": n, "threshold": r.count,
                                         "track_ids": sorted(o.track_id for o in in_zone)}))

            # Forget per-track state for tracks that are gone (bounded memory).
            if len(st.seen) > 4096:
                st.seen &= live_ids
            for tid in [t for t in st.last_point if t not in live_ids]:
                del st.last_point[tid]
            for key in [k for k, v in st.last_fired.items()
                        if ts - v > max(r.cooldown_s, 1.0) and _track_of(k) is not None and _track_of(k) not in live_ids]:
                del st.last_fired[key]
        return events


def _track_of(key) -> Optional[int]:
    """The track id inside a dedup key (zone_dwell: tid, line_cross: (tid, dir)), else None."""
    if isinstance(key, int):
        return key
    if isinstance(key, tuple) and key and isinstance(key[0], int):
        return key[0]
    return None


def rule_by_name(config: CameraConfig, name: str) -> Optional[Rule]:
    return next((r for r in config.rules if r.name == name), None)


def as_array(objects: List[TrackedObject]) -> np.ndarray:
    return np.array([o.box for o in objects], dtype=np.float32).reshape(-1, 4)
