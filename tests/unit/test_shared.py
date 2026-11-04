"""shared/: letterbox geometry, decoding, gate metrics, camera config validation."""

import numpy as np
import pytest
from pydantic import ValidationError

from yolo_platform_shared.camera_config import CameraConfig
from yolo_platform_shared.decode import box_iou, decode_classification, decode_detections
from yolo_platform_shared.gate import evaluate_classify, evaluate_detect, match
from yolo_platform_shared.preprocess import LetterboxMeta, classify_crop, letterbox, letterbox_params, scale_boxes


@pytest.mark.parametrize("w,h", [(640, 480), (1920, 1080), (360, 640), (640, 640), (1281, 719), (100, 50)])
def test_letterbox_shape_padding_and_rgb(w, h):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = 10   # B
    img[..., 2] = 200  # R
    out, meta = letterbox(img, 640)
    assert out.shape == (640, 640, 3) and out.dtype == np.uint8
    new_w, new_h, top, bottom, left, right, r = letterbox_params(w, h, 640)
    assert top + new_h + bottom == 640 and left + new_w + right == 640
    assert (meta.pad_left, meta.pad_top, meta.ratio) == (left, top, r)
    # Content is RGB (R first), padding is 114.
    assert tuple(out[top + new_h // 2, left + new_w // 2]) == (200, 0, 10)
    if top:
        assert tuple(out[0, 320]) == (114, 114, 114)
    if left:
        assert tuple(out[320, 0]) == (114, 114, 114)


def test_letterbox_into_preallocated_view():
    batch = np.zeros((2, 640, 640, 3), dtype=np.uint8)
    img = np.full((480, 640, 3), 7, dtype=np.uint8)
    out, _ = letterbox(img, 640, out=batch[1])
    assert out.base is batch or np.shares_memory(out, batch)
    assert batch[1, 320, 320, 0] == 7 and batch[0].max() == 0


def test_scale_boxes_roundtrip():
    meta = LetterboxMeta(orig_w=1920, orig_h=1080, ratio=640 / 1920, pad_left=0, pad_top=140)
    orig = np.array([[100, 200, 500, 900], [0, 0, 1920, 1080]], dtype=np.float32)
    model_space = orig * meta.ratio
    model_space[:, [1, 3]] += meta.pad_top
    np.testing.assert_allclose(scale_boxes(model_space, meta), orig, atol=1e-3)


def test_scale_boxes_clips_to_image():
    meta = LetterboxMeta(orig_w=100, orig_h=100, ratio=6.4, pad_left=0, pad_top=0)
    out = scale_boxes(np.array([[-10, -10, 700, 700]], dtype=np.float32), meta)
    np.testing.assert_allclose(out, [[0, 0, 100, 100]])


def test_classify_crop_centre_crops_short_side():
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    img[:, 100:200] = 255
    out = classify_crop(img, 224)
    assert out.shape == (224, 224, 3)
    assert out.mean() > 250  # the centre third is white: the crop keeps only it


def _raw(rows, max_det=300):
    out = np.zeros((max_det, 6), dtype=np.float32)
    out[:len(rows)] = rows
    return out


def test_decode_detections_threshold_classes_and_mapping():
    meta = LetterboxMeta(orig_w=1280, orig_h=720, ratio=0.5, pad_left=0, pad_top=140)
    raw = _raw([[100, 240, 200, 340, 0.9, 0], [10, 150, 20, 160, 0.2, 0], [300, 300, 400, 400, 0.6, 2]])
    names = {0: "person", 2: "car"}
    dets = decode_detections(raw, meta, names, conf=0.25)
    assert [d.class_name for d in dets] == ["person", "car"]
    assert (dets[0].x1, dets[0].y1, dets[0].x2, dets[0].y2) == pytest.approx((200, 200, 400, 400))
    assert [d.class_name for d in decode_detections(raw, meta, names, conf=0.25, classes=[2])] == ["car"]
    d = dets[0].to_dict()
    assert d["bbox"] == {"x1": 200.0, "y1": 200.0, "x2": 400.0, "y2": 400.0} and d["confidence"] == 0.9


def test_decode_classification_topk_sorted():
    probs = np.array([0.1, 0.6, 0.05, 0.25], dtype=np.float32)
    top = decode_classification(probs, {0: "a", 1: "b", 2: "c", 3: "d"}, top_k=2)
    assert [(c.class_name, round(c.confidence, 2)) for c in top] == [("b", 0.6), ("d", 0.25)]
    assert len(decode_classification(probs, {}, top_k=50)) == 4


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=np.float32)
    np.testing.assert_allclose(box_iou(a, b), [[1.0, 1 / 3, 0.0]], atol=1e-6)
    assert box_iou(a, np.zeros((0, 4))).shape == (1, 0)


def test_gate_match_respects_class_and_is_one_to_one():
    ref = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    got = np.array([[0, 0, 10, 10]], dtype=np.float32)
    pairs = match(ref, np.array([0, 1]), got, np.array([1]))
    assert pairs == [(1, 0, 1.0)]


def _golden_detect():
    boxes = np.array([[10, 10, 100, 100], [200, 200, 300, 300], [400, 400, 450, 450]], dtype=np.float32)
    return {"ref_boxes": boxes, "ref_conf": np.array([0.9, 0.8, 0.5], dtype=np.float32),
            "ref_cls": np.array([0, 2, 0]), "ref_img": np.array([0, 0, 1])}


def test_gate_detect_passes_on_identical_outputs():
    g = _golden_detect()
    outs = np.zeros((2, 300, 6), dtype=np.float32)
    outs[0, :2, :4], outs[0, :2, 4], outs[0, :2, 5] = g["ref_boxes"][:2], g["ref_conf"][:2], g["ref_cls"][:2]
    outs[1, 0, :4], outs[1, 0, 4], outs[1, 0, 5] = g["ref_boxes"][2], 0.5, 0
    r = evaluate_detect(g, outs)
    assert r["passed"] and r["recall"] == 1.0 and r["precision"] == 1.0 and r["mean_iou"] == 1.0


def test_gate_detect_fails_on_wrong_class_or_missing_boxes():
    g = _golden_detect()
    outs = np.zeros((2, 300, 6), dtype=np.float32)
    outs[0, :2, :4], outs[0, :2, 4], outs[0, :2, 5] = g["ref_boxes"][:2], g["ref_conf"][:2], 5  # wrong class
    r = evaluate_detect(g, outs)
    assert not r["passed"] and r["recall"] == 0.0


def test_gate_detect_fails_with_no_reference_boxes():
    g = {"ref_boxes": np.zeros((0, 4), np.float32), "ref_conf": np.zeros(0, np.float32),
         "ref_cls": np.zeros(0, int), "ref_img": np.zeros(0, int)}
    assert not evaluate_detect(g, np.zeros((1, 300, 6), np.float32))["passed"]


def test_gate_classify():
    g = {"ref_top1": np.array([1, 0]), "ref_top1_prob": np.array([0.9, 0.7], dtype=np.float32)}
    good = np.array([[0.1, 0.9], [0.71, 0.29]], dtype=np.float32)
    assert evaluate_classify(g, good)["passed"]
    bad = np.array([[0.9, 0.1], [0.71, 0.29]], dtype=np.float32)
    assert not evaluate_classify(g, bad)["passed"]


def test_camera_config_valid_and_models_used():
    cfg = CameraConfig.model_validate({
        "rules": [{"name": "door", "type": "zone_dwell", "zone": [[0, 0], [0.5, 0], [0.5, 1]],
                   "classify": {"model": "yolo26n_cls"}},
                  {"name": "gate", "type": "line_cross", "line": [[0, 0.5], [1, 0.5]]}],
        "triggered": [{"model": "ppe_detection", "on_rule": "door"}, {"model": "fire_detection", "every_s": 1}],
    })
    assert cfg.models_used() == ["fire_detection", "ppe_detection", "yolo26n_cls"]


@pytest.mark.parametrize("bad", [
    {"rules": [{"name": "z", "type": "zone_dwell"}]},                                        # zone missing
    {"rules": [{"name": "z", "type": "presence", "zone": [[0, 0], [2, 0], [1, 1]]}]},        # not normalised
    {"rules": [{"name": "l", "type": "line_cross", "line": [[0.1, 0.1], [0.1, 0.1]]}]},      # degenerate line
    {"rules": [{"name": "a", "type": "presence"}, {"name": "a", "type": "presence"}]},       # duplicate names
    {"triggered": [{"model": "fire_detection"}]},                                            # no trigger
    {"triggered": [{"model": "fire_detection", "every_s": 1, "on_rule": "x"}]},              # both triggers
    {"triggered": [{"model": "ppe_detection", "on_rule": "nope"}]},                          # unknown rule
])
def test_camera_config_rejects(bad):
    with pytest.raises(ValidationError):
        CameraConfig.model_validate(bad)
