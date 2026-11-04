"""GPU letterbox kernel vs the CPU reference (needs a GPU; skipped otherwise)."""

import numpy as np
import pytest

from yolo_platform_shared.preprocess import letterbox

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
    cp.zeros(1)
except Exception:  # no GPU / driver in this container
    pytest.skip("no CUDA device", allow_module_level=True)

from workers.gpu_prep import letterbox_gpu  # noqa: E402


@pytest.mark.parametrize("w,h", [(640, 480), (1920, 1080), (480, 360), (360, 640), (640, 640)])
def test_gpu_letterbox_matches_cpu(w, h):
    rng = np.random.default_rng(w * h)
    # Smooth content: bilinear rounding differences stay at +-1 grey level.
    base = rng.integers(0, 255, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    import cv2
    rgb = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    bgr = rgb[..., ::-1].copy()
    cpu, cpu_meta = letterbox(bgr, 640)
    dst = cp.empty((640, 640, 3), dtype=cp.uint8)
    gpu_meta = letterbox_gpu(cp.asarray(rgb), 640, dst)       # NVDEC hands over RGB
    gpu = cp.asnumpy(dst)
    assert gpu_meta == cpu_meta
    diff = np.abs(gpu.astype(int) - cpu.astype(int))
    assert diff.max() <= 2, diff.max()
    assert diff.mean() < 0.3


def test_gpu_letterbox_bgr_input_and_pitched_source():
    rgb = np.zeros((360, 480, 3), dtype=np.uint8)
    rgb[..., 0] = 250
    padded = cp.zeros((360, 512, 3), dtype=cp.uint8)           # row pitch > width
    padded[:, :480] = cp.asarray(rgb[..., ::-1])               # stored as BGR
    dst = cp.empty((640, 640, 3), dtype=cp.uint8)
    letterbox_gpu(padded[:, :480], 640, dst, src_is_bgr=True)
    centre = cp.asnumpy(dst[320, 320])
    assert tuple(centre) == (250, 0, 0)
