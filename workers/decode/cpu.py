"""CPU decode (OpenCV/FFmpeg). The fallback and the benchmark baseline.

The reader drains every frame with grab() (which decodes) and only retrieves and
letterboxes the frames that are due. This is the "drain thread" pattern measured
in lost-found: without it a slow consumer lets the RTSP buffer fill and frames are
lost in bursts.
"""

import os
import time

import cv2
import numpy as np

from yolo_platform_shared.preprocess import letterbox

from .base import FrameSource

# TCP for RTSP, 10 s socket timeout (microseconds) so a dead camera raises instead of hanging.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|timeout;10000000")


class CpuSource(FrameSource):
    backend = "cpu"

    def __init__(self, camera_id: str, url: str, target_fps: float, size: int) -> None:
        super().__init__(camera_id, url, target_fps, size)
        self._bufs = self._alloc(1)
        self._cap = None

    def _alloc(self, windows: int) -> list:
        return [np.empty((windows, self.size, self.size, 3), dtype=np.uint8) for _ in range(2)]

    def _stream(self) -> None:
        url = self.url[len("file://"):] if self.url.startswith("file://") else self.url
        self._cap = cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ConnectionError("could not open stream")
        self.stats.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.stats.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        self.stats.codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00") if fourcc else ""
        # Files are paced to their frame rate, so they behave like a live camera.
        pace = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25.0) if self.url.startswith("file://") else 0.0
        frame = None
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                if not cap.grab():
                    raise ConnectionError("read failed")
                now = time.time()
                self.stats.decoded += 1
                self.stats.last_frame_at = now
                self._mark_streaming()
                if self._due(now):
                    ok, frame = cap.retrieve(frame)
                    if ok:
                        windows = self._windows(frame.shape[1], frame.shape[0])
                        buf = self._back_buffer(len(windows))
                        metas = [letterbox(frame[y1:y2, x1:x2], self.size, out=buf[k])[1]
                                 for k, (x1, y1, x2, y2) in enumerate(windows)]
                        self._publish(metas, now, windows)
                if pace:
                    left = pace - (time.monotonic() - t0)
                    if left > 0:
                        time.sleep(left)
        finally:
            cap.release()
            self._cap = None

    def _copy_out(self, index: int, dst) -> None:
        dst[...] = self._bufs[index][:len(dst)]
