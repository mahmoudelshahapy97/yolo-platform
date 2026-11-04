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


# NV12 (what NVDEC produces natively) -> letterboxed RGB, fused. Converting only the
# sampled frames (instead of asking the decoder for RGB on every decoded frame) costs
# a third of the CPU time per decoded frame (measured: 0.22 vs 0.65 ms on a T4 host).
_NV12_KERNEL = cp.RawKernel(r"""
__device__ __forceinline__ float bilerp(const unsigned char* p, long pitch, int stride, int off,
                                        float fx, float fy, int w, int h) {
    fx = fminf(fmaxf(fx, 0.0f), (float)(w - 1));
    fy = fminf(fmaxf(fy, 0.0f), (float)(h - 1));
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    float ax = fx - x0, ay = fy - y0;
    const unsigned char* r0 = p + (long)y0 * pitch;
    const unsigned char* r1 = p + (long)y1 * pitch;
    return (1 - ay) * ((1 - ax) * r0[x0 * stride + off] + ax * r0[x1 * stride + off])
         +      ay  * ((1 - ax) * r1[x0 * stride + off] + ax * r1[x1 * stride + off]);
}

extern "C" __global__
void nv12_letterbox(const unsigned char* __restrict__ y_plane, long y_pitch,
                    const unsigned char* __restrict__ uv_plane, long uv_pitch,
                    int src_w, int src_h, unsigned char* __restrict__ dst, int size,
                    int new_w, int new_h, int left, int top, float sx, float sy,
                    float ky, float y_off, float kr, float kgu, float kgv, float kb)
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
    float Y = bilerp(y_plane, y_pitch, 1, 0, fx, fy, src_w, src_h);
    float cx = (fx + 0.5f) * 0.5f - 0.5f, cy = (fy + 0.5f) * 0.5f - 0.5f;
    int cw = (src_w + 1) / 2, ch = (src_h + 1) / 2;
    float U = bilerp(uv_plane, uv_pitch, 2, 0, cx, cy, cw, ch) - 128.0f;
    float V = bilerp(uv_plane, uv_pitch, 2, 1, cx, cy, cw, ch) - 128.0f;
    float yy = (Y - y_off) * ky;
    out[0] = (unsigned char)fminf(fmaxf(yy + kr * V + 0.5f, 0.0f), 255.0f);
    out[1] = (unsigned char)fminf(fmaxf(yy - kgu * U - kgv * V + 0.5f, 0.0f), 255.0f);
    out[2] = (unsigned char)fminf(fmaxf(yy + kb * U + 0.5f, 0.0f), 255.0f);
}
""", "nv12_letterbox")


def yuv_coefficients(bt709: bool, full_range: bool):
    """(ky, y_offset, kr, kgu, kgv, kb) for Y'CbCr -> R'G'B'."""
    kr_, kb_ = (0.2126, 0.0722) if bt709 else (0.299, 0.114)
    kg_ = 1 - kr_ - kb_
    if full_range:
        ky, y_off, cs = 1.0, 0.0, 1.0
    else:
        ky, y_off, cs = 255.0 / 219.0, 16.0, 255.0 / 224.0
    kr = 2 * (1 - kr_) * cs
    kb = 2 * (1 - kb_) * cs
    kgu = 2 * kb_ * (1 - kb_) / kg_ * cs
    kgv = 2 * kr_ * (1 - kr_) / kg_ * cs
    return ky, y_off, kr, kgu, kgv, kb


def nv12_letterbox_gpu(y_ptr: int, y_pitch: int, uv_ptr: int, uv_pitch: int, w: int, h: int, size: int,
                       dst: cp.ndarray, stream: cp.cuda.Stream | None = None, bt709: bool = False,
                       full_range: bool = False) -> LetterboxMeta:
    """NV12 planes at raw device pointers -> dst [size, size, 3] RGB uint8, letterboxed."""
    new_w, new_h, top, _bottom, left, _right, r = letterbox_params(w, h, size)
    ky, y_off, kr, kgu, kgv, kb = yuv_coefficients(bt709, full_range)
    block = (16, 16, 1)
    grid = ((size + 15) // 16, (size + 15) // 16, 1)
    args = (cp.uint64(y_ptr), cp.int64(y_pitch), cp.uint64(uv_ptr), cp.int64(uv_pitch), cp.int32(w), cp.int32(h),
            dst, cp.int32(size), cp.int32(new_w), cp.int32(new_h), cp.int32(left), cp.int32(top),
            cp.float32(w / new_w), cp.float32(h / new_h), cp.float32(ky), cp.float32(y_off), cp.float32(kr),
            cp.float32(kgu), cp.float32(kgv), cp.float32(kb))
    if stream is not None:
        with stream:
            _NV12_KERNEL(grid, block, args)
    else:
        _NV12_KERNEL(grid, block, args)
    return LetterboxMeta(orig_w=w, orig_h=h, ratio=r, pad_left=left, pad_top=top)
