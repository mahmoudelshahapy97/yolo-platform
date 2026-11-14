"""Tiled cameras: config, multi-window samples, slot filling and merged decoding
(no Triton, no GPU). The CPU source test decodes a generated 1080p file."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from yolo_platform_shared.camera_config import CameraConfig, Tiling
from yolo_platform_shared.decode import Detection, merge_windows
from yolo_platform_shared.preprocess import LetterboxMeta, letterbox_params
from yolo_platform_shared.tiling import TileConfig, plan_windows

pytest.importorskip("tritonclient")

from workers import camera_shard as cs  # noqa: E402
from workers.decode.base import FrameSource, Sample  # noqa: E402

HD = plan_windows(1920, 1080, TileConfig())  # whole frame + 2x2


def meta_for(window, size=640):
    x1, y1, x2, y2 = window
    new_w, new_h, top, _, left, _, r = letterbox_params(x2 - x1, y2 - y1, size)
    return LetterboxMeta(orig_w=x2 - x1, orig_h=y2 - y1, ratio=r, pad_left=left, pad_top=top)


def model_row(window, box, conf=0.9, cls=0, size=640):
    """A detection at frame-pixel `box`, as the engine reports it for `window`."""
    m = meta_for(window, size)
    x1, y1 = window[:2]
    return [(box[0] - x1) * m.ratio + m.pad_left, (box[1] - y1) * m.ratio + m.pad_top,
            (box[2] - x1) * m.ratio + m.pad_left, (box[3] - y1) * m.ratio + m.pad_top, conf, cls]


def outputs(rows_per_window, cols=6):
    out = np.zeros((len(rows_per_window), 300, cols), dtype=np.float32)
    for k, rows in enumerate(rows_per_window):
        for i, r in enumerate(rows):
            out[k, i, :len(r)] = r
    return out


# --------------------------------------------------------------------- config
def test_cameras_default_to_auto_tiling():
    cfg = CameraConfig().tiling.tile_config()
    assert (cfg.mode, cfg.grid, cfg.min_side) == ("auto", (2, 2), 960)
    assert CameraConfig.model_validate({"tiling": {"mode": "off"}}).tiling.mode == "off"
    assert CameraConfig.model_validate({"tiling": {"grid": "3X2"}}).tiling.grid == "3x2"


@pytest.mark.parametrize("bad", [{"mode": "sometimes"}, {"grid": "4x4"}, {"grid": "two"}, {"overlap": 0.7},
                                 {"merge_iou": 1.0}, {"min_side": -5}])
def test_bad_tiling_is_rejected(bad):
    with pytest.raises(ValidationError):
        Tiling.model_validate(bad)


# -------------------------------------------------------------------- samples
class ArraySource(FrameSource):
    """A FrameSource whose 'decoder' is the test: it writes windows and publishes."""

    def __init__(self, size=4):
        super().__init__("cam", "unused", 5.0, size)
        self._bufs = self._alloc(1)

    def _alloc(self, windows):
        return [np.zeros((windows, self.size, self.size, 3), dtype=np.uint8) for _ in range(2)]

    def _copy_out(self, index, dst):
        dst[...] = self._bufs[index][:len(dst)]

    def decode(self, width=1920, height=1080):
        windows = self._windows(width, height)
        buf = self._back_buffer(len(windows))
        for k in range(len(windows)):
            buf[k] = k + 1
        self._publish([meta_for(w) for w in windows], time.time(), windows)


def test_untiled_samples_are_one_window():
    src = ArraySource()
    src.decode()
    assert src.fresh_windows() == 1
    dst = np.zeros((16, 4, 4, 3), dtype=np.uint8)
    sample = src.take(dst)
    assert sample.windows == ((0, 0, 1920, 1080),) and not sample.tiled and sample.meta == sample.metas[0]
    assert dst[0].max() == 1 and dst[1].max() == 0


def test_tiled_samples_fill_one_slot_per_window():
    src = ArraySource()
    src.set_tiling(TileConfig())
    src.decode()
    assert src.fresh_windows() == 5
    dst = np.zeros((16, 4, 4, 3), dtype=np.uint8)
    sample = src.take(dst)
    assert sample.tiled and sample.windows == tuple(HD) and sample.meta.orig_w == 1920
    assert [int(dst[k].max()) for k in range(6)] == [1, 2, 3, 4, 5, 0]
    assert sample.metas[1].orig_w == 1067 and src.fresh_windows() == 0


def test_a_sample_that_does_not_fit_stays_fresh():
    src = ArraySource()
    src.set_tiling(TileConfig())
    src.decode()
    assert src.take(np.zeros((3, 4, 4, 3), dtype=np.uint8)) is None
    assert src.fresh_windows() == 5


def test_substreams_stay_one_window_under_auto():
    src = ArraySource()
    src.set_tiling(TileConfig())
    src.decode(640, 360)
    assert src.fresh_windows() == 1


def test_growing_the_buffers_drops_the_untaken_sample():
    src = ArraySource()
    src.decode()
    src.set_tiling(TileConfig(grid=(3, 2)))
    src.decode()
    assert src.stats.dropped == 1 and src.fresh_windows() == 7


# -------------------------------------------------------------------- decoding
NAMES = {0: "person", 1: "bag"}


def sample_for(windows):
    metas = tuple(meta_for(w) for w in windows)
    return Sample(1, time.time(), metas[0], metas, tuple(windows))


def test_tiled_outputs_merge_into_frame_pixels():
    sample = sample_for(HD)
    bag = (1500, 900, 1520, 915)      # too small for the whole frame: only the tile sees it
    person = (300, 200, 380, 400)     # seen whole and in the top-left tile
    out = outputs([[model_row(HD[0], person, 0.8)], [model_row(HD[1], person, 0.9)], [], [],
                   [model_row(HD[4], bag, 0.6, cls=1)]])
    dets = cs.Shard._decode(sample, out, {"task": "detect", "names": NAMES}, 0.1, None, 0.5)
    got = sorted(((d.class_name, round(d.x1), round(d.y1), round(d.x2), round(d.y2)) for d in dets))
    assert got == [("bag", *bag), ("person", *person)]
    assert max(d.confidence for d in dets if d.class_name == "person") == pytest.approx(0.9)


def test_untiled_decode_is_unchanged():
    sample = sample_for(HD[:1])
    out = outputs([[model_row(HD[0], (10, 20, 110, 220))]])
    dets = cs.Shard._decode(sample, out, {"task": "detect", "names": NAMES}, 0.1, None, 0.5)
    assert [(round(d.x1), round(d.y1), round(d.x2), round(d.y2)) for d in dets] == [(10, 20, 110, 220)]


def test_merged_keypoints_move_with_their_box():
    kp = np.array([[10.0, 20.0, 0.9], [30.0, 40.0, 0.8]], dtype=np.float32)
    d = Detection(0, "person", 0.9, 10, 10, 50, 90, keypoints=kp)
    merged = merge_windows([[], [], [], [], [d]], HD)
    x1, y1 = HD[4][:2]
    assert merged[0].x1 == 10 + x1 and merged[0].keypoints[:, :2].tolist() == [[10 + x1, 20 + y1], [30 + x1, 40 + y1]]
    assert d.keypoints[0, 0] == 10.0  # the window's own record is untouched


# ------------------------------------------------------------------ pipeline
class FakeSlot:
    def __init__(self):
        self.inputs = np.zeros((cs.SLOT_BATCH, 1, 1, 3), dtype=np.uint8)
        self.calls = []

    def infer(self, name, version, n, out_cols=None, start=0):
        self.calls.append((start, n))
        return None

    def output(self, n, out_cols=None):
        return np.zeros((n, 300, out_cols or 6), dtype=np.float32)

    def close(self):
        pass


class WindowSource:
    """Always has a fresh sample of `k` windows."""

    def __init__(self, k):
        self.k = k
        self.taken = 0

    def has_fresh(self):
        return True

    def fresh_windows(self):
        return self.k

    def take(self, dst):
        if self.k > len(dst):
            return None
        self.taken += 1
        return sample_for([(0, 0, 1920, 1080)] + [(0, 0, 960, 540)] * (self.k - 1))


def run_rounds(monkeypatch, cams, rounds=1, max_batch=8):
    """Run the pipeline for `rounds` rounds; returns the slot's (start, n) requests and
    what _post got per sample: (camera, windows, outputs, frames)."""
    sh = cs.Shard.__new__(cs.Shard)
    sh.s = SimpleNamespace(triton_url="unused")
    sh.models = SimpleNamespace(get=lambda name: {"task": "detect", "imgsz": 640, "max_batch": max_batch,
                                                  "serve_name": "m", "names": NAMES})
    sh.cam_lock = threading.Lock()
    sh.cameras = {c.id: c for c in cams}
    sh.stop = threading.Event()
    sh._rr = 0
    sh._unservable_warned = {}
    slot = FakeSlot()
    posted = []
    monkeypatch.setattr(cs.grpcclient, "InferenceServerClient", lambda url: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(sh, "_new_slot", lambda client, i: slot, raising=False)
    monkeypatch.setattr(sh, "_post", lambda rt, sample, outs, frames, model, client:
                        posted.append((rt.id, len(sample.windows), len(outs), len(frames))), raising=False)
    n = {"i": 0}
    real = sh._fresh_by_model

    def fresh():
        n["i"] += 1
        if n["i"] > rounds:
            sh.stop.set()
            return {}
        return real()
    monkeypatch.setattr(sh, "_fresh_by_model", fresh, raising=False)
    sh._pipeline(0)
    return slot.calls, posted


def cam(cam_id, k):
    return SimpleNamespace(id=cam_id, spec={"model": "m"}, model="m", source=WindowSource(k))


def test_untiled_round_is_one_request_as_before(monkeypatch):
    calls, posted = run_rounds(monkeypatch, [cam(f"c{i}", 1) for i in range(10)])
    assert calls == [(0, 8)] and len(posted) == 8


def test_tiled_and_plain_cameras_share_a_request(monkeypatch):
    calls, posted = run_rounds(monkeypatch, [cam("tiled", 5), cam("a", 1), cam("b", 1), cam("c", 1)])
    assert calls == [(0, 8)]
    assert sorted(p[1] for p in posted) == [1, 1, 1, 5] and all(p[1] == p[2] == p[3] for p in posted)


def test_a_sample_bigger_than_max_batch_goes_alone_in_chunks_and_is_not_starved(monkeypatch):
    calls, posted = run_rounds(monkeypatch, [cam("big", 13), cam("a", 1)], rounds=4)
    assert ("big", 13, 13, 13) in posted and ("a", 1, 1, 1) in posted
    i = calls.index((0, 8))
    assert calls[i + 1] == (8, 5)  # the rest of the same sample, right after
    assert all(n <= 8 for _, n in calls)


# ------------------------------------------------------------------ CPU source
def test_cpu_source_letterboxes_each_window(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from workers.decode.cpu import CpuSource

    path = tmp_path / "hd.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (1920, 1080))
    if not writer.isOpened():
        pytest.skip("no mp4 writer")
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[700:1050, 1500:1900] = 255  # white block in the bottom-right tile only
    for _ in range(30):
        writer.write(frame)
    writer.release()

    src = CpuSource("cam", f"file://{path}", 50.0, 640)
    src.set_tiling(TileConfig())
    src.start()
    try:
        deadline = time.time() + 10
        while src.fresh_windows() == 0 and time.time() < deadline:
            time.sleep(0.02)
        dst = np.zeros((16, 640, 640, 3), dtype=np.uint8)
        sample = src.take(dst)
    finally:
        src.stop()
        src.join(5)
    assert sample is not None and sample.windows == tuple(HD)
    assert [m.orig_w for m in sample.metas] == [1920, 1067, 1067, 1067, 1067]
    # White shows in the whole frame and the bottom-right tile, not the top-left tile.
    assert dst[0].max() > 200 and dst[4].max() > 200 and dst[1].max() < 140
