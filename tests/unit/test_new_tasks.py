"""shared/: pose and segmentation decoding, their gates, the sequence gate, and shoplifting windows."""

import numpy as np
import pytest

from yolo_platform_shared.behavior import (NUM_FEATURES, NUM_FRAMES, NUM_KEYPOINTS, Person, ShopliftingParams,
                                           ShopliftingTracker, extract_features, keypoints_xyn, pad_window,
                                           windows_from_sequence)
from yolo_platform_shared.camera_config import CameraConfig
from yolo_platform_shared.decode import Detection, decode_pose, decode_segment, full_masks
from yolo_platform_shared.gate import evaluate, evaluate_pose, evaluate_segment, evaluate_sequence, sample_masks
from yolo_platform_shared.preprocess import LetterboxMeta

META = LetterboxMeta(orig_w=1280, orig_h=720, ratio=0.5, pad_left=0, pad_top=140)


def pose_row(box, conf, kpts):
    return np.concatenate([np.array([*box, conf, 0], np.float32), np.asarray(kpts, np.float32).reshape(-1)])


def test_decode_pose_maps_keypoints_to_original_pixels():
    kpts = np.zeros((17, 3), np.float32)
    kpts[:, 0], kpts[:, 1], kpts[:, 2] = 100, 240, 0.9   # model space
    out = np.zeros((300, 57), np.float32)
    out[0] = pose_row((50, 200, 150, 400), 0.8, kpts)
    out[1] = pose_row((0, 0, 10, 10), 0.1, kpts)          # below threshold
    dets = decode_pose(out, META, {0: "person"}, (17, 3), conf=0.25)
    assert len(dets) == 1
    kp = dets[0].keypoints
    assert kp.shape == (17, 3)
    np.testing.assert_allclose(kp[0], [200, 200, 0.9], rtol=1e-5)  # (100-0)/0.5, (240-140)/0.5
    d = dets[0].to_dict()
    assert len(d["keypoints"]) == 17 and "polygon" not in d


def test_detection_to_dict_omits_absent_extras():
    d = Detection(0, "person", 0.5, 1, 2, 3, 4).to_dict()
    assert set(d) == {"class_id", "class_name", "confidence", "bbox"}


def _rect_protos(x0, y0, x1, y1, size=160):
    """32 prototypes; prototype 0 is +8 inside the rectangle (proto pixels), -8 outside."""
    protos = np.zeros((32, size, size), np.float32)
    protos[0] = -8
    protos[0, y0:y1, x0:x1] = 8
    return protos


def test_decode_segment_polygon_follows_the_mask_and_the_box():
    protos = _rect_protos(40, 60, 80, 100)            # model space 160..320 x 240..400
    out = np.zeros((300, 38), np.float32)
    out[0, :6] = [100, 200, 400, 600, 0.9, 0]         # the box is larger than the mask
    out[0, 6] = 1.0
    dets = decode_segment(out, protos, META, {0: "thing"}, 640, conf=0.25)
    poly = dets[0].polygon
    assert poly.shape[1] == 2 and len(poly) >= 4
    # Mask 160..320 x 240..400 model space -> original: x (160..320)/0.5 = 320..640, y (240-140..400-140)/0.5 = 200..520
    assert abs(poly[:, 0].min() - 320) <= 4 and abs(poly[:, 0].max() - 640) <= 4
    assert abs(poly[:, 1].min() - 200) <= 4 and abs(poly[:, 1].max() - 520) <= 4
    # Cropped to the box: a box that cuts the mask cuts the polygon.
    out[0, :4] = [100, 200, 240, 600]
    poly = decode_segment(out, protos, META, {0: "thing"}, 640, conf=0.25)[0].polygon
    assert poly[:, 0].max() <= 240 / 0.5 + 1


def test_full_masks_match_decode_segment_area():
    protos = _rect_protos(40, 60, 80, 100)
    boxes = np.array([[100, 200, 400, 600]], np.float32)
    coeffs = np.zeros((1, 32), np.float32)
    coeffs[0, 0] = 1
    m = full_masks(coeffs, protos, boxes, 640)[0]
    ys, xs = np.nonzero(m)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (160, 319, 240, 399)


def _pose_golden():
    kpts = np.random.default_rng(0).uniform(100, 500, (17, 3)).astype(np.float32)
    kpts[:, 2] = 0.9
    out = np.zeros((1, 300, 57), np.float32)
    out[0, 0] = pose_row((100, 100, 300, 500), 0.9, kpts)
    golden = {"ref_boxes": np.array([[100, 100, 300, 500]], np.float32), "ref_conf": np.array([0.9], np.float32),
              "ref_cls": np.array([0]), "ref_img": np.array([0]), "ref_kpts": kpts[None]}
    return golden, out


def test_gate_pose_passes_identical_and_fails_on_shifted_keypoints():
    golden, out = _pose_golden()
    assert evaluate_pose(golden, out)["passed"]
    shifted = out.copy()
    shifted[0, 0, 6::3] += 10
    r = evaluate_pose(golden, shifted)
    assert not r["passed"] and r["mean_kpt_err_px"] == pytest.approx(10, abs=0.01)


def test_gate_segment_passes_identical_and_fails_on_wrong_mask():
    protos = _rect_protos(40, 60, 80, 100)[None]
    out = np.zeros((1, 300, 38), np.float32)
    out[0, 0, :6] = [100, 200, 400, 600, 0.9, 0]
    out[0, 0, 6] = 1.0
    ref = sample_masks(full_masks(out[0, :1, 6:], protos[0], out[0, :1, :4], 640))
    golden = {"ref_boxes": out[0, :1, :4].copy(), "ref_conf": np.array([0.9], np.float32), "ref_cls": np.array([0]),
              "ref_img": np.array([0]), "ref_masks": ref, "inputs": np.zeros((1, 640, 640, 3), np.uint8)}
    r = evaluate("segment", golden, [out, protos], {"imgsz": 640})
    assert r["passed"] and r["mean_mask_iou"] == 1.0
    wrong = out.copy()
    wrong[0, 0, 6] = -1.0   # the complement: the box minus the rectangle
    assert not evaluate_segment(golden, wrong, protos, 640)["passed"]


def test_gate_sequence_is_stricter_than_classify():
    golden = {"ref_top1": np.array([0, 1]), "ref_top1_prob": np.array([0.9, 0.8], np.float32)}
    assert evaluate_sequence(golden, np.array([[0.9, 0.1], [0.2, 0.8]], np.float32))["passed"]
    r = evaluate_sequence(golden, np.array([[0.86, 0.14], [0.24, 0.76]], np.float32))   # 0.04 off
    assert not r["passed"]


# ---------------------------------------------------------------- behaviour


def kp(x=0.5, y=0.5):
    k = np.zeros((17, 2), np.float32)
    k[:, 0], k[:, 1] = x, y
    k[9] = (x - 0.05, y)       # left wrist
    k[10] = (x + 0.05, y)      # right wrist
    k[11] = (x - 0.02, y + 0.1)
    k[12] = (x + 0.02, y + 0.1)
    return k


def test_extract_features_layout():
    f = extract_features(kp(0.5, 0.5), kp(0.4, 0.5))
    assert f.shape == (NUM_KEYPOINTS, NUM_FEATURES) and f.dtype == np.float32
    np.testing.assert_allclose(f[0, :4], [0.5, 0.5, 0.1, 0.0], atol=1e-6)
    mid_hip = np.array([0.5, 0.6])
    np.testing.assert_allclose(f[3, 4:7], [np.linalg.norm([0.45, 0.5] - mid_hip),
                                           np.linalg.norm([0.55, 0.5] - mid_hip), 0.1], atol=1e-6)
    assert (f[:, 7:] == 0).all()


def test_keypoints_xyn_and_window_padding():
    px = np.zeros((17, 3), np.float32)
    px[:, 0], px[:, 1] = 640, 360
    np.testing.assert_allclose(keypoints_xyn(px, 1280, 720), 0.5)
    w = windows_from_sequence([kp(), None, kp(0.6)])
    assert w.shape == (NUM_FRAMES, NUM_KEYPOINTS, NUM_FEATURES)
    assert (w[:-3] == 0).all() and (w[-2] == 0).all()
    assert (w[-1, :, 2:4] == 0).all()        # no velocity across the gap
    assert pad_window([]).shape == (NUM_FRAMES, NUM_KEYPOINTS, NUM_FEATURES)


def person(tid, x=0.5, height=200.0):
    return Person(tid, (100.0, 100.0, 160.0, 100.0 + height), kp(x))


def test_tracker_samples_at_rate_and_waits_for_enough_samples():
    t = ShopliftingTracker(ShopliftingParams(sample_fps=2.0, min_window_samples=3))
    due = []
    for i in range(12):                       # 5 fps processing, 2 fps sampling
        due += [(i, tid) for tid, _ in t.update([person(1)], i * 0.2)]
    # Due every 0.5 s with 25 % slack: samples at 0, 0.4, 0.8, 1.2, 1.6, 2.0.
    assert len(t.tracks[1].window) == 6
    assert due and due[0][0] == 4             # first judgement on the 3rd real sample (t = 0.8)


def test_tracker_window_equals_offline_window():
    t = ShopliftingTracker(ShopliftingParams(sample_fps=1.0, min_window_samples=1))
    seq = [kp(0.3 + 0.01 * i) for i in range(5)]
    last = None
    for i, k in enumerate(seq):
        out = t.update([Person(7, (0, 0, 10, 300), k)], float(i))
        last = out[-1][1]
    np.testing.assert_allclose(last, windows_from_sequence(seq))


def test_tracker_smoothing_threshold_and_cooldowns():
    p = ShopliftingParams(sample_fps=1.0, min_window_samples=1, smooth_window=2, threshold=0.5,
                          track_cooldown_s=100, camera_cooldown_s=10)
    t = ShopliftingTracker(p)
    t.update([person(1), person(2, x=0.2)], 0.0)
    assert t.judge(1, 0.9, 0.0) is not None              # mean of [0.9]
    assert t.judge(1, 0.9, 1.0) is None                  # track cooldown
    assert t.judge(2, 0.95, 1.0) is None                 # camera cooldown (10 s)
    a = t.judge(2, 0.95, 11.0)
    assert a is not None and a.track_id == 2 and a.probability == pytest.approx(0.95)
    t2 = ShopliftingTracker(p)
    t2.update([person(1)], 0.0)
    assert t2.judge(1, 0.9, 0.0) is not None
    t2b = ShopliftingTracker(p)
    t2b.update([person(3)], 0.0)
    assert t2b.judge(3, 0.2, 0.0) is None and t2b.judge(3, 0.7, 1.0) is None   # smoothed 0.45 < 0.5
    assert t2b.judge(99, 0.99, 1.0) is None                                     # unknown / gone track


def test_tracker_handover_keeps_window_and_expires_lost_tracks():
    t = ShopliftingTracker(ShopliftingParams(sample_fps=1.0, handover_s=5.0, track_max_misses=3))
    for i in range(4):
        t.update([person(1)], float(i))
    t.update([], 4.0)                                     # lost: an absent row
    t.update([person(9)], 5.0)                            # new id, similar height -> takes over
    assert 1 not in t.tracks and len(t.tracks[9].window) == 6
    t.update([person(20, height=600)], 6.0)               # very different height: a new person
    assert 20 in t.tracks and len(t.tracks[20].window) == 1
    for i in range(5):
        t.update([person(20, height=600)], 7.0 + i)
    assert 9 not in t.tracks                              # expired after track_max_misses


def test_camera_config_shoplifting():
    c = CameraConfig.model_validate({"shoplifting": {"threshold": 0.15}})
    assert c.shoplifting.model == "shoplifting_seq" and c.shoplifting.sample_fps == 3.0
    assert "shoplifting_seq" in c.models_used()
    with pytest.raises(Exception):
        CameraConfig.model_validate({"shoplifting": {"min_window_samples": 500}})


def test_duplicate_engine_boxes_are_suppressed_per_class():
    from yolo_platform_shared.decode import decode_detections, select_rows
    out = np.zeros((300, 6), np.float32)
    out[0] = [100, 100, 200, 200, 0.86, 66]
    out[1] = [101, 101, 201, 199, 0.83, 66]   # FP16 duplicate (IoU ~0.97)
    out[2] = [101, 101, 201, 199, 0.80, 1]    # same place, other class: kept
    out[3] = [300, 300, 400, 400, 0.50, 66]
    rows = select_rows(out, 0.25, None, 300)
    assert rows[:, 4].tolist() == pytest.approx([0.86, 0.80, 0.50])
    assert len(decode_detections(out, META, {}, conf=0.25)) == 3
