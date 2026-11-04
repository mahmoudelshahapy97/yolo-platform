"""Letterbox on the GPU: decoded RGB frame (device) -> padded model input (device), in one kernel.

Same geometry as yolo_platform_shared.preprocess.letterbox (Ultralytics' rounding),
with bilinear sampling on OpenCV's half-pixel grid, so CPU and GPU paths agree to
within interpolation rounding (tests/test_gpu_prep.py checks it).
"""

import cupy as cp

from yolo_platform_shared.preprocess import LetterboxMeta, letterbox_params

_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void letterbox_rgb(const unsigned char* __restrict__ src, int src_w, int src_h, long src_pitch,
                   unsigned char* __restrict__ dst, int size,
                   int new_w, int new_h, int left, int top, float sx, float sy, int swap_rb)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= size || y >= size) return;
    unsigned char* out = dst + ((long)y * size + x) * 3;
    int ix = x - left, iy = y - top;
    if (ix < 0 || iy < 0 || ix >= new_w || iy >= new_h) {
        out[0] = 114; out[1] = 114; out[2] = 114;
        return;
    }
    float fx = (ix + 0.5f) * sx - 0.5f;
    float fy = (iy + 0.5f) * sy - 0.5f;
    fx = fminf(fmaxf(fx, 0.0f), (float)(src_w - 1));
    fy = fminf(fmaxf(fy, 0.0f), (float)(src_h - 1));
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    float ax = fx - x0, ay = fy - y0;
    const unsigned char* r0 = src + (long)y0 * src_pitch;
    const unsigned char* r1 = src + (long)y1 * src_pitch;
    for (int c = 0; c < 3; ++c) {
        int sc = swap_rb ? 2 - c : c;
        float v = (1 - ay) * ((1 - ax) * r0[x0 * 3 + sc] + ax * r0[x1 * 3 + sc])
                +      ay  * ((1 - ax) * r1[x0 * 3 + sc] + ax * r1[x1 * 3 + sc]);
        out[c] = (unsigned char)fminf(fmaxf(v + 0.5f, 0.0f), 255.0f);
    }
}
""", "letterbox_rgb")


def letterbox_gpu(src: cp.ndarray, size: int, dst: cp.ndarray, stream: cp.cuda.Stream | None = None,
                  src_is_bgr: bool = False) -> LetterboxMeta:
    """src: device uint8 [H, W, 3] (row pitch may exceed W*3); dst: device uint8 [size, size, 3] contiguous."""
    h, w = src.shape[:2]
    new_w, new_h, top, _bottom, left, _right, r = letterbox_params(w, h, size)
    block = (16, 16, 1)
    grid = ((size + 15) // 16, (size + 15) // 16, 1)
    args = (src, cp.int32(w), cp.int32(h), cp.int64(src.strides[0]), dst, cp.int32(size),
            cp.int32(new_w), cp.int32(new_h), cp.int32(left), cp.int32(top),
            cp.float32(w / new_w), cp.float32(h / new_h), cp.int32(1 if src_is_bgr else 0))
    if stream is not None:
        with stream:
            _KERNEL(grid, block, args)
    else:
        _KERNEL(grid, block, args)
    return LetterboxMeta(orig_w=w, orig_h=h, ratio=r, pad_left=left, pad_top=top)
