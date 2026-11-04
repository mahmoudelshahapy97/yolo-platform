"""Our letterbox / box mapping / classify crop vs Ultralytics' own (runs in the export image).

    docker run --rm -v $PWD:/work -w /work yolo-platform-export:dev \
        sh -c "pip install -q pytest==8.4.2 && PYTHONPATH=shared python -m pytest -q tests/export"
"""

import numpy as np
import pytest
import torch
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.utils import ops

from yolo_platform_shared.preprocess import LetterboxMeta, classify_crop, letterbox, scale_boxes


@pytest.mark.parametrize("w,h", [(640, 480), (1920, 1080), (480, 360), (360, 640), (1281, 719), (2560, 1440)])
def test_letterbox_identical_to_ultralytics(w, h):
    img = np.random.default_rng(w + h).integers(0, 255, (h, w, 3), dtype=np.uint8)
    ref = LetterBox((640, 640), auto=False, stride=32)(image=img)       # BGR, as predict() does
    ours, _ = letterbox(img, 640)
    np.testing.assert_array_equal(ours, ref[..., ::-1])                  # ours is RGB


@pytest.mark.parametrize("w,h", [(640, 480), (1920, 1080), (360, 640)])
def test_scale_boxes_identical_to_ultralytics(w, h):
    _, meta = letterbox(np.zeros((h, w, 3), np.uint8), 640)
    boxes = np.array([[10, 20, 300, 400], [0, 0, 640, 640], [100.5, 200.25, 101, 300]], dtype=np.float32)
    ref = ops.scale_boxes((640, 640), torch.tensor(boxes.copy()), (h, w)).numpy()
    np.testing.assert_allclose(scale_boxes(boxes, meta), ref, atol=0.5)


def test_classify_crop_close_to_ultralytics():
    img = np.random.default_rng(3).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    import cv2
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)
    from PIL import Image
    ref = classify_transforms(224)(Image.fromarray(img[..., ::-1]))      # CHW float in [0, 1], RGB
    ref = (ref.permute(1, 2, 0).numpy() * 255).round()
    ours = classify_crop(img, 224).astype(np.float32)
    assert ours.shape == ref.shape
    assert np.abs(ours - ref).mean() < 3.0                               # antialias filter differences only
