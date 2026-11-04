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


@pytest.mark.parametrize("w,h", [(640, 360), (540, 360), (1920, 1080), (480, 360)])
def test_nv12_letterbox_matches_opencv_conversion(w, h):
    import cv2

    from workers.gpu_prep import nv12_letterbox_gpu

    rng = np.random.default_rng(w + h)
    base = rng.integers(0, 255, (h // 16 + 1, w // 16 + 1, 3), dtype=np.uint8)
    rgb = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    i420 = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)                   # BT.601 limited range
    y = i420[:h]
    u = i420[h:h + h // 4].reshape(h // 2, w // 2)
    v = i420[h + h // 4:].reshape(h // 2, w // 2)
    nv12 = np.concatenate([y, np.stack([u, v], axis=-1).reshape(h // 2, w)], axis=0)
    ref_bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    ref, ref_meta = letterbox(ref_bgr, 640)

    # Pitched planes, as NVDEC hands them over.
    pitch = (w + 255) // 256 * 256
    dev = cp.zeros((h + h // 2, pitch), dtype=cp.uint8)
    dev[:, :w] = cp.asarray(nv12)
    dst = cp.empty((640, 640, 3), dtype=cp.uint8)
    meta = nv12_letterbox_gpu(dev.data.ptr, pitch, dev.data.ptr + h * pitch, pitch, w, h, 640, dst)
    out = cp.asnumpy(dst)
    assert meta == ref_meta
    # Judge both conversions against the true RGB image: OpenCV upsamples chroma
    # nearest-neighbour and converts before resizing, the kernel interpolates in YUV
    # with bilinear chroma. The GPU path must be at least as accurate, and unbiased
    # per channel (a wrong matrix or swapped channel would show up here).
    truth, _ = letterbox(np.ascontiguousarray(rgb[..., ::-1]), 640)
    gpu_err = np.abs(out.astype(int) - truth.astype(int))
    cv_err = np.abs(ref.astype(int) - truth.astype(int))
    assert gpu_err.mean() <= cv_err.mean() + 0.25, (gpu_err.mean(), cv_err.mean())
    bias = (out.astype(int) - truth.astype(int)).reshape(-1, 3).mean(axis=0)
    assert np.abs(bias).max() < 1.0, bias
