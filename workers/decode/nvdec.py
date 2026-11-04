"""NVDEC decode with letterboxing on the GPU.

Demux: PyAV (FFmpeg) in-process, because it takes FFmpeg options: RTSP over TCP
and socket timeouts, so a dead camera raises instead of hanging. (PyNvVideoCodec's
own demuxer takes no options and only tries RTSP over UDP.) The H.264/HEVC
access units are converted to Annex-B (prepending SPS/PPS from the SDP extradata
on keyframes for RTSP; the mp4toannexb filter for MP4 files) and handed to an
NVDEC session.

Every source in a shard shares the shard's one CUDA context (the primary
context, owned by CuPy), so a camera costs a decoder session and a few MB of
VRAM, not a CUDA context of its own. Decoded frames stay on the GPU: due frames
are letterboxed by a CUDA kernel into the camera's device slot, and only the
640x640 model input crosses to the host (into the Triton shared-memory slot).
"""

import threading
import time

import av
import cupy as cp
import numpy as np
import PyNvVideoCodec as nvc
from cupy_backends.cuda.api import driver

from ..gpu_prep import letterbox_gpu
from .base import FrameSource

_CTX_LOCK = threading.Lock()
_CTX = None
CODECS = {"h264": nvc.cudaVideoCodec.H264, "hevc": nvc.cudaVideoCodec.HEVC}
OPEN_OPTIONS = {"rtsp_transport": "tcp", "timeout": "10000000", "rw_timeout": "10000000", "fflags": "nobuffer"}


def shard_context() -> int:
    """The CUDA primary context, created once per process by CuPy."""
    global _CTX
    with _CTX_LOCK:
        if _CTX is None:
            cp.cuda.Device(0).use()
            cp.zeros(1)  # forces context creation
            _CTX = driver.ctxGetCurrent()
        return _CTX


def frame_as_cupy(frame) -> cp.ndarray:
    """Zero-copy view of a decoded RGB frame (valid until the decoder reuses the surface)."""
    cai = frame.cuda()[0].__cuda_array_interface__
    shape, strides = tuple(cai["shape"]), tuple(cai["strides"])
    mem = cp.cuda.UnownedMemory(cai["data"][0], shape[0] * strides[0], owner=frame)
    return cp.ndarray(shape, cp.uint8, cp.cuda.MemoryPointer(mem, 0), strides=strides)


class AnnexB:
    """Turns demuxed packets into Annex-B access units NVDEC can parse."""

    def __init__(self, stream) -> None:
        name = stream.codec_context.name
        extradata = bytes(stream.codec_context.extradata or b"")
        # avcC / hvcC extradata starts with version byte 1 (MP4-style length-prefixed NALs).
        self.bsf = av.bitstream.BitStreamFilterContext(f"{name}_mp4toannexb", stream) if extradata[:1] == b"\x01" else None
        self.headers = extradata if self.bsf is None else b""

    def __call__(self, packet):
        if self.bsf is not None:
            for p in self.bsf.filter(packet):
                yield bytes(p), p.pts
        else:
            data = bytes(packet)
            if packet.is_keyframe and self.headers:
                data = self.headers + data
            yield data, packet.pts


class NvdecSource(FrameSource):
    backend = "nvdec"

    def __init__(self, camera_id: str, url: str, target_fps: float, size: int) -> None:
        super().__init__(camera_id, url, target_fps, size)
        self._ctx = shard_context()
        cp.cuda.Device(0).use()
        self._stream_obj = cp.cuda.Stream(non_blocking=True)
        self._bufs = [cp.empty((size, size, 3), dtype=cp.uint8) for _ in range(2)]

    def _stream(self) -> None:
        cp.cuda.Device(0).use()
        is_file = self.url.startswith("file://")
        url = self.url[len("file://"):] if is_file else self.url
        container = av.open(url, options={} if is_file else OPEN_OPTIONS, timeout=(10.0, 10.0))
        try:
            vs = container.streams.video[0]
            name = vs.codec_context.name
            if name not in CODECS:
                raise ValueError(f"codec {name} is not supported by the NVDEC backend (h264, hevc)")
            self.stats.width, self.stats.height, self.stats.codec = vs.codec_context.width, vs.codec_context.height, name
            to_annexb = AnnexB(vs)
            decoder = nvc.CreateDecoder(gpuid=0, codec=CODECS[name], cudacontext=self._ctx,
                                        cudastream=self._stream_obj.ptr, usedevicememory=True,
                                        outputColorType=nvc.OutputColorType.RGB,
                                        latency=nvc.DisplayDecodeLatencyType.LOW)
            pace = 1.0 / float(vs.average_rate or 25) if is_file else 0.0
            got_any = False
            for packet in container.demux(vs):
                if self._stop.is_set():
                    return
                if packet.size == 0:
                    continue
                t0 = time.monotonic()
                for data, pts in to_annexb(packet):
                    buf = np.frombuffer(data, dtype=np.uint8)
                    pd = nvc.PacketData()
                    pd.bsl_data, pd.bsl, pd.pts = buf.ctypes.data, buf.size, pts or 0
                    for frame in decoder.Decode(pd):
                        now = time.time()
                        got_any = True
                        self.stats.decoded += 1
                        self.stats.last_frame_at = now
                        self._mark_streaming()
                        if self._due(now):
                            meta = letterbox_gpu(frame_as_cupy(frame), self.size, self._bufs[self.back],
                                                 self._stream_obj)
                            self._stream_obj.synchronize()   # the surface is reused by the next Decode
                            self._publish(meta, now)
                if pace:
                    left = pace - (time.monotonic() - t0)
                    if left > 0:
                        time.sleep(left)
            if not got_any:
                raise ConnectionError("no frames decoded")
        finally:
            container.close()

    def _copy_out(self, index: int, dst) -> None:
        # dst is a slot of the shard's shared-memory batch buffer (host). One D2H copy
        # of the 640x640 input is all that crosses PCIe; the full frame never does.
        self._bufs[index].get(out=dst)
