"""workers/rules.py and workers/tracking.py with synthetic detections (no model, no I/O)."""

import numpy as np

from workers.rules import RulesEngine, TrackedObject, point_in_polygon, segments_intersect
from workers.tracking import ByteTracker
from yolo_platform_shared.camera_config import CameraConfig

W, H = 1000, 1000


def obj(tid, x, y, cls="person", conf=0.9, size=50):
    """An object standing at (x, y) in pixels (bottom-centre anchor)."""
    return TrackedObject(tid, cls, conf, (x - size / 2, y - 2 * size, x + size / 2, y))


def engine(*rules):
    return RulesEngine(CameraConfig.model_validate({"rules": list(rules)}))


def test_geometry_helpers():
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert point_in_polygon(0.5, 0.5, sq) and not point_in_polygon(1.5, 0.5, sq)
    assert segments_intersect((0, 0), (1, 1), (0, 1), (1, 0))
    assert not segments_intersect((0, 0), (0.4, 0.4), (0, 1), (1, 0))


def test_presence_once_per_track_with_class_cooldown():
    e = engine({"name": "p", "type": "presence", "cooldown_s": 10})
    assert len(e.evaluate([obj(1, 100, 100)], W, H, 0.0)) == 1
    assert e.evaluate([obj(1, 110, 100)], W, H, 1.0) == []          # same track: never again
    assert e.evaluate([obj(2, 300, 100)], W, H, 2.0) == []          # new track, within cooldown
    assert len(e.evaluate([obj(3, 300, 100)], W, H, 12.5)) == 1     # after cooldown


def test_presence_filters_class_confidence_and_zone():
    e = engine({"name": "p", "type": "presence", "classes": ["car"], "min_confidence": 0.5, "cooldown_s": 0,
                "zone": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]})
    evs = e.evaluate([obj(1, 100, 100, "person"), obj(2, 100, 100, "car", 0.3), obj(3, 800, 100, "car"),
                      obj(4, 100, 500, "car")], W, H, 0.0)
    assert [ev.track_id for ev in evs] == [4]


def test_zone_dwell_fires_after_dwell_and_resets_on_exit():
    e = engine({"name": "d", "type": "zone_dwell", "zone": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]], "dwell_s": 3,
                "cooldown_s": 100})
    assert e.evaluate([obj(1, 100, 500)], W, H, 0.0) == []
    assert e.evaluate([obj(1, 100, 500)], W, H, 2.9) == []
    evs = e.evaluate([obj(1, 100, 500)], W, H, 3.1)
    assert len(evs) == 1 and evs[0].data["dwell_s"] >= 3
    assert e.evaluate([obj(1, 100, 500)], W, H, 10) == []            # cooldown per track
    e.evaluate([obj(1, 900, 500)], W, H, 11)                          # leaves the zone
    assert e.evaluate([obj(1, 100, 500)], W, H, 12) == []             # dwell restarts


def test_line_cross_direction_and_dedup():
    e = engine({"name": "l", "type": "line_cross", "line": [[0.5, 0], [0.5, 1]], "cooldown_s": 100})
    assert e.evaluate([obj(1, 400, 500)], W, H, 0) == []
    evs = e.evaluate([obj(1, 600, 500)], W, H, 1)
    assert len(evs) == 1
    first = evs[0].data["direction"]
    back = e.evaluate([obj(1, 400, 500)], W, H, 2)
    assert len(back) == 1 and back[0].data["direction"] != first
    assert e.evaluate([obj(1, 600, 500)], W, H, 3) == []              # same direction again: cooldown


def test_count_above():
    e = engine({"name": "c", "type": "count_above", "count": 2, "cooldown_s": 5})
    two = [obj(1, 100, 100), obj(2, 200, 100)]
    assert e.evaluate(two, W, H, 0) == []
    evs = e.evaluate(two + [obj(3, 300, 100)], W, H, 1)
    assert len(evs) == 1 and evs[0].data["count"] == 3 and evs[0].data["track_ids"] == [1, 2, 3]
    assert e.evaluate(two + [obj(3, 300, 100)], W, H, 2) == []


def _box(x, y, s=40):
    return [x, y, x + s, y + 2 * s]


def test_bytetrack_keeps_ids_for_moving_objects():
    t = ByteTracker(frame_rate=5)
    ids_a, ids_b = set(), set()
    for f in range(20):
        boxes = np.array([_box(100 + 5 * f, 100), _box(500 - 5 * f, 300)], dtype=np.float32)
        out = dict(t.update(boxes, np.array([0.9, 0.85]), np.array([0, 0])))
        if f >= 1:
            assert set(out) == {0, 1}
            ids_a.add(out[0])
            ids_b.add(out[1])
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_bytetrack_low_confidence_keeps_track_alive_and_recovers_after_gap():
    t = ByteTracker(frame_rate=5, buffer_s=2)
    for f in range(3):
        out = t.update(np.array([_box(100 + 4 * f, 100)]), np.array([0.9]), np.array([0]))
    tid = out[0][1]
    out = t.update(np.array([_box(112, 100)]), np.array([0.2]), np.array([0]))     # occluded: low score
    assert out == [(0, tid)]
    for _ in range(4):                                                              # missing for 4 frames
        assert t.update(np.zeros((0, 4)), np.zeros(0), np.zeros(0)) == []
    out = t.update(np.array([_box(132, 100)]), np.array([0.9]), np.array([0]))
    assert out == [(0, tid)]


def test_bytetrack_classes_do_not_match_each_other():
    t = ByteTracker(frame_rate=5)
    for _ in range(2):
        first = dict(t.update(np.array([_box(100, 100)]), np.array([0.9]), np.array([0])))
    t.update(np.array([_box(100, 100)]), np.array([0.9]), np.array([2]))            # same place, other class
    out = dict(t.update(np.array([_box(100, 100)]), np.array([0.9]), np.array([2])))
    assert out and out[0] != first[0]


def test_bytetrack_drops_lost_tracks_after_buffer():
    t = ByteTracker(frame_rate=5, buffer_s=1)
    for _ in range(3):
        t.update(np.array([_box(100, 100)]), np.array([0.9]), np.array([0]))
    for _ in range(7):
        t.update(np.zeros((0, 4)), np.zeros(0), np.zeros(0))
    assert t.tracks == []
