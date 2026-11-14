"""Tiled (SAHI-style) inference: the whole frame plus overlapping tiles, merged.

A detector sees a frame letterboxed to its input size, so a 1080p frame at 640
px shrinks every object to a third; people far from a high camera, and small
items, fall below what the model can find. Running the model on overlapping
tiles as well magnifies them (2x2 tiles of a 1080p frame: about 1.8x), and the
whole-frame pass still sees objects too big for one tile.

    windows = plan_windows(w, h, cfg)           # whole frame first, then tiles
    dets = [predict(frame[y1:y2, x1:x2]) for (x1, y1, x2, y2) in windows]
    rows = merge(dets, windows, iou=cfg.merge_iou, agnostic=True)

Per-window detections are rows ``[x1, y1, x2, y2, ..., conf, cls]`` in that
window's own pixels (what a detector returns for the crop); merge() returns rows
of the same layout in frame pixels. Pure numpy: the same file is used by
lost-found, InsightEye and yolo-platform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

Window = Tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels, x2/y2 exclusive

MODES = ("off", "auto", "on")
# Box edges this close to a tile's inner edge mean the object runs out of the tile.
EDGE_PX = 3
# A box cut by a tile edge is dropped when a box of the same class, seen whole in
# another window, holds at least this much of it.
CUT_COVERED = 0.6
# Same-class boxes from different windows are one object when the smaller lies
# this much inside the larger (a part seen in a tile, the whole in the frame).
NESTED_IOS = 0.8


@dataclass(frozen=True)
class TileConfig:
    """How a camera's frames are tiled. ``auto`` tiles only frames whose long side
    is at least ``min_side``: substreams (720x480) stay one pass, 1080p is tiled."""

    mode: str = "auto"
    grid: Tuple[int, int] = (2, 2)  # columns, rows
    overlap: float = 0.2
    min_side: int = 960
    merge_iou: float = 0.5

    # Keys accepted in a camera's settings, and the field each one sets.
    KEYS = {"tile_mode": "mode", "tile_grid": "grid", "tile_overlap": "overlap",
            "tile_min_side": "min_side", "tile_merge_iou": "merge_iou"}

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"tile_mode must be one of {', '.join(MODES)}, not {self.mode!r}")
        cols, rows = self.grid
        if not (1 <= cols <= 6 and 1 <= rows <= 6):
            raise ValueError(f"tile_grid must be 1x1 to 6x6, not {cols}x{rows}")
        if not 0.0 <= self.overlap <= 0.5:
            raise ValueError("tile_overlap must be between 0 and 0.5")
        if not 0 <= self.min_side <= 16384:
            raise ValueError("tile_min_side must be between 0 and 16384")
        if not 0.05 <= self.merge_iou <= 0.95:
            raise ValueError("tile_merge_iou must be between 0.05 and 0.95")

    def with_overrides(self, settings: Optional[Mapping[str, Any]]) -> "TileConfig":
        """This config with a camera's ``tile_*`` settings applied. Raises ValueError
        on a bad value; other keys are ignored."""
        if not settings:
            return self
        changes = {}
        for key, field_name in self.KEYS.items():
            if key not in settings or settings[key] is None:
                continue
            value = settings[key]
            try:
                if field_name == "mode":
                    value = str(value).strip().lower()
                elif field_name == "grid":
                    value = parse_grid(value)
                elif field_name == "min_side":
                    if isinstance(value, bool):
                        raise TypeError
                    value = int(value)
                else:
                    if isinstance(value, bool):
                        raise TypeError
                    value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key} has an invalid value: {settings[key]!r}") from None
            changes[field_name] = value
        return replace(self, **changes) if changes else self

    def tiles(self, width: int, height: int) -> bool:
        """Whether a frame of this size is tiled."""
        if self.mode == "off" or self.grid == (1, 1):
            return False
        return self.mode == "on" or max(width, height) >= self.min_side


def parse_grid(value: Any) -> Tuple[int, int]:
    """'2x2', '3x2', (3, 2) or [3, 2] -> (columns, rows)."""
    if isinstance(value, str):
        parts = value.lower().replace("*", "x").split("x")
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError(f"grid must look like '2x2', not {value!r}")
    if len(parts) != 2:
        raise ValueError(f"grid must look like '2x2', not {value!r}")
    cols, rows = (int(str(p).strip()) for p in parts)
    return cols, rows


def _spans(length: int, count: int, overlap: float) -> List[Tuple[int, int]]:
    """`count` equal spans covering [0, length) with `overlap` between neighbours,
    the last one flush with the end."""
    if count <= 1 or length <= 1:
        return [(0, length)]
    # The epsilon keeps 1080 / 1.8 at 600, not 601 from float error.
    size = min(length, math.ceil(length / (count - (count - 1) * overlap) - 1e-9))
    step = (length - size) / (count - 1)
    return [(round(i * step), round(i * step) + size) for i in range(count)]


def plan_windows(width: int, height: int, cfg: TileConfig) -> List[Window]:
    """The windows to run the detector on: the whole frame, then the tiles if this
    frame is tiled. All tiles have the same size, so they batch together."""
    full = (0, 0, int(width), int(height))
    if not cfg.tiles(width, height):
        return [full]
    cols, rows = cfg.grid
    tiles = [(x1, y1, x2, y2)
             for y1, y2 in _spans(int(height), rows, cfg.overlap)
             for x1, x2 in _spans(int(width), cols, cfg.overlap)]
    return [full] + [t for t in tiles if t != full]


def crops(frame: np.ndarray, windows: Sequence[Window]) -> List[np.ndarray]:
    """Views of the frame for each window (no copies)."""
    return [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in windows]


def _area(b: np.ndarray) -> np.ndarray:
    return np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)


def _inter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection areas of boxes a [N,4] and b [M,4] -> [N,M]."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(rb - lt, 0, None)
    return wh[..., 0] * wh[..., 1]


def merge(dets_per_window: Sequence[Optional[np.ndarray]], windows: Sequence[Window],
          iou: float = 0.5, agnostic: bool = True, max_det: int = 300) -> np.ndarray:
    """Merge per-window detections into one set in frame pixels.

    1. Shift each window's boxes into frame pixels.
    2. Drop a tile box cut by one of the tile's inner edges when a same-class box
       seen whole in another window covers it (the object is in another view).
    3. Greedy NMS by confidence over pairs from different windows: IoU >= `iou`
       suppresses (across classes when `agnostic`), and a same-class box lying
       >= 80% inside a kept one is a duplicate. Pairs from the same window are left
       to the model's own NMS, and a wallet in a person's hand is a different class.
    """
    if len(dets_per_window) != len(windows):
        raise ValueError(f"{len(dets_per_window)} detection sets for {len(windows)} windows")
    width = max(w[2] for w in windows) if windows else 0
    height = max(w[3] for w in windows) if windows else 0

    rows, origin, cut = [], [], []
    cols = None
    for index, (dets, (x1, y1, x2, y2)) in enumerate(zip(dets_per_window, windows)):
        if dets is None or len(dets) == 0:
            continue
        d = np.asarray(dets, dtype=np.float32).copy()
        if cols is None:
            cols = d.shape[1]
        elif d.shape[1] != cols:
            raise ValueError(f"window {index} has {d.shape[1]} columns, expected {cols}")
        d[:, [0, 2]] += x1
        d[:, [1, 3]] += y1
        # Inner edges only: the frame's own border cuts nothing.
        edge = np.zeros(len(d), dtype=bool)
        if x1 > 0:
            edge |= d[:, 0] <= x1 + EDGE_PX
        if x2 < width:
            edge |= d[:, 2] >= x2 - EDGE_PX
        if y1 > 0:
            edge |= d[:, 1] <= y1 + EDGE_PX
        if y2 < height:
            edge |= d[:, 3] >= y2 - EDGE_PX
        rows.append(d)
        origin.append(np.full(len(d), index))
        cut.append(edge)

    if not rows:
        return np.zeros((0, cols or 6), dtype=np.float32)
    boxes = np.concatenate(rows)
    origin = np.concatenate(origin)
    cut = np.concatenate(cut)
    conf, cls = boxes[:, -2], boxes[:, -1].astype(np.int64)

    inter = _inter(boxes[:, :4], boxes[:, :4])
    area = _area(boxes[:, :4])
    same_class = cls[:, None] == cls[None, :]
    other_window = origin[:, None] != origin[None, :]

    # 2. A cut box another window saw whole.
    if cut.any():
        covered = (inter / np.maximum(area[:, None], 1e-6)) >= CUT_COVERED  # [i, j]: j holds i
        whole = ~cut[None, :] & same_class & other_window & covered
        keep = ~(cut & whole.any(axis=1))
        boxes, origin, conf, cls = boxes[keep], origin[keep], conf[keep], cls[keep]
        inter, area = inter[keep][:, keep], area[keep]
        same_class, other_window = same_class[keep][:, keep], other_window[keep][:, keep]

    # 3. Greedy NMS.
    union = area[:, None] + area[None, :] - inter
    iou_m = inter / np.maximum(union, 1e-6)
    ios_m = inter / np.maximum(np.minimum(area[:, None], area[None, :]), 1e-6)
    overlap = iou_m >= iou
    if not agnostic:
        overlap &= same_class
    overlap |= same_class & (ios_m >= NESTED_IOS)
    # Within one window the model's own NMS has decided already (two people
    # overlapping in a crowd stay two); only views of the same spot are merged.
    overlap &= other_window

    order = np.argsort(-conf, kind="stable")
    suppressed = np.zeros(len(boxes), dtype=bool)
    kept = []
    for i in order:
        if suppressed[i]:
            continue
        kept.append(i)
        if len(kept) >= max_det:
            break
        suppressed |= overlap[i]
    return boxes[np.array(kept, dtype=np.int64)]
