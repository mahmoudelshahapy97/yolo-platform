"""Frame sources: one thread per camera that decodes continuously and keeps only
the latest *sampled* frame, already letterboxed to the detector's input size.

Latest frame wins. A sample the batcher has not taken yet is overwritten
by the next one and counted as dropped, so a slow GPU never builds a backlog of
stale frames. Decoding never stops between samples: H.264/HEVC need every frame,
and falling behind a live stream means the server buffers and then drops us.

Each source double-buffers its letterboxed slot. ``take(dst)`` copies the
fresh sample into the caller's batch buffer under the slot lock, so the decoder
can never overwrite a frame while it is being batched.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from yolo_platform_shared.preprocess import LetterboxMeta

logger = logging.getLogger(__name__)


@dataclass
class SourceStats:
    state: str = "starting"           # starting | streaming | reconnecting | stopped | failed
    error: Optional[str] = None
    width: int = 0
    height: int = 0
    codec: str = ""
    decoded: int = 0                  # frames decoded
    sampled: int = 0                  # frames letterboxed into the slot
    taken: int = 0                    # samples handed to the batcher
    dropped: int = 0                  # samples overwritten before the batcher took them
    reconnects: int = 0
    last_frame_at: float = 0.0        # wall clock of the latest decoded frame
    started_at: float = field(default_factory=time.time)


@dataclass
class Sample:
    seq: int
    ts: float                         # wall clock when the frame was decoded
    meta: LetterboxMeta


class OpenThrottle:
    """Process-wide spacing between stream opens, so 40 cameras starting together
    do not hit the network, the cameras and the decoder all at once."""

    def __init__(self, spacing_s: float = 0.25) -> None:
        self.spacing_s = spacing_s
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self, stop: threading.Event) -> bool:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.spacing_s
        delay = slot - time.monotonic()
        return not stop.wait(delay) if delay > 0 else not stop.is_set()


OPEN_THROTTLE = OpenThrottle()


class FrameSource:
    """Base class. Subclasses implement ``_stream`` (open, decode until error or stop)."""

    backend = "base"

    def __init__(self, camera_id: str, url: str, target_fps: float, size: int) -> None:
        self.camera_id = camera_id
        self.url = url
        self.size = size
        self.target_fps = target_fps
        self.stats = SourceStats()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._fresh: Optional[Sample] = None
        self._front = 0                 # index of the buffer holding the fresh sample
        self._seq = 0
        self._next_sample = 0.0
        self._thread: Optional[threading.Thread] = None

    # ----- lifecycle -----

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"src-{self.camera_id[:8]}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float) -> bool:
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def set_fps(self, fps: float) -> None:
        self.target_fps = fps

    def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            if not OPEN_THROTTLE.wait(self._stop):
                break
            try:
                self.stats.state = "starting" if attempt == 0 else "reconnecting"
                self._stream()
                if self._stop.is_set():
                    break
                raise ConnectionError("stream ended")
            except Exception as exc:  # any failure: back off and reopen
                if self._stop.is_set():
                    break
                attempt += 1
                self.stats.reconnects += 1 if attempt > 1 else 0
                self.stats.state = "reconnecting"
                self.stats.error = f"{type(exc).__name__}: {exc}"[:300]
                delay = min(30.0, 1.0 * 2 ** min(attempt - 1, 5))
                logger.warning("camera %s: %s; reopening in %.0fs", self.camera_id, self.stats.error, delay)
                self._stop.wait(delay)
            else:
                attempt = 0
        self.stats.state = "stopped"
        self._close()

    def _mark_streaming(self) -> None:
        if self.stats.state != "streaming":
            self.stats.state = "streaming"
            self.stats.error = None

    # ----- sampling -----

    def _due(self, now: float) -> bool:
        """True when this decoded frame should be letterboxed (target_fps on the wall clock)."""
        if now < self._next_sample:
            return False
        period = 1.0 / max(self.target_fps, 0.01)
        # Keep a steady cadence, but never try to "catch up" after a stall.
        self._next_sample = max(self._next_sample + period, now + period * 0.5) if self._next_sample else now + period
        return True

    def _publish(self, meta: LetterboxMeta, now: float) -> None:
        """Called by the decoder thread after writing the back buffer."""
        with self._lock:
            if self._fresh is not None:
                self.stats.dropped += 1
            self._seq += 1
            self._front ^= 1
            self._fresh = Sample(self._seq, now, meta)
        self.stats.sampled += 1

    @property
    def back(self) -> int:
        """Index of the buffer the decoder may write (never the fresh one)."""
        return self._front ^ 1

    def take(self, dst) -> Optional[Sample]:
        """Copy the fresh sample into ``dst`` (host or device array) and mark it taken."""
        with self._lock:
            sample = self._fresh
            if sample is None:
                return None
            self._copy_out(self._front, dst)
            self._fresh = None
        self.stats.taken += 1
        return sample

    def has_fresh(self) -> bool:
        return self._fresh is not None

    # ----- subclass hooks -----

    def _stream(self) -> None:
        raise NotImplementedError

    def _copy_out(self, index: int, dst) -> None:
        raise NotImplementedError

    def _close(self) -> None:
        pass
