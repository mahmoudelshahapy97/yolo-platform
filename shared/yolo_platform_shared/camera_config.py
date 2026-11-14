"""Per-camera configuration: rules and triggered models. Validated by the API, read by the shards.

Coordinates in zones and lines are normalised to [0, 1] of the frame, so a rule
survives a change of stream resolution (main stream vs substream).

Rule types:
  presence    a tracked object of ``classes`` is seen (once per track, then cooldown)
  zone_dwell  a tracked object stays inside ``zone`` for >= ``dwell_s`` seconds
  line_cross  a tracked object's bottom-centre crosses ``line`` (direction in data)
  count_above more than ``count`` objects of ``classes`` are visible at once (inside ``zone`` if given)

Triggered models run on top of the primary detector, so not every model runs
on every frame:
  every_s     run periodically on the latest frame (e.g. fire detection once a second)
  on_rule     run when the named rule fires (e.g. PPE check when a person enters a zone)
A rule's ``classify`` runs a classifier on the crop of the object that fired it.

``shoplifting`` (pose camera models only) keeps a window of every tracked
person's pose and judges it with a sequence model (``shoplifting_seq``); see
behavior.py. Alerts are events of type ``shoplifting``.

``tiling`` runs the camera's primary model on overlapping tiles as well as the
whole frame and merges the boxes (tiling.py), so far-away people and small
objects are seen magnified. ``auto`` (the default) tiles only frames whose long
side is at least ``min_side``: a 1080p main stream is tiled (2x2: 5 inferences
per frame), a 640x360 substream is not. ``off`` restores one pass per frame.
"""

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from .tiling import TileConfig, parse_grid

Point = Tuple[float, float]

# A tiled frame's windows (whole frame + tiles) must fit one shard batch slot.
MAX_TILE_WINDOWS = 16


class Classify(BaseModel):
    model: str = "yolo26n_cls"
    top_k: int = Field(1, ge=1, le=10)


class Rule(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[\w\-. ]+$")
    type: Literal["presence", "zone_dwell", "line_cross", "count_above"]
    classes: List[str] = Field(default_factory=lambda: ["person"])
    min_confidence: float = Field(0.4, ge=0.0, le=1.0)
    zone: Optional[List[Point]] = None
    line: Optional[Tuple[Point, Point]] = None
    dwell_s: float = Field(5.0, ge=0.0, le=3600)
    count: int = Field(1, ge=0, le=1000)
    cooldown_s: float = Field(30.0, ge=0.0, le=86400)
    classify: Optional[Classify] = None

    @field_validator("zone")
    @classmethod
    def _zone(cls, v):
        if v is None:
            return v
        if len(v) < 3:
            raise ValueError("a zone needs at least 3 points")
        for x, y in v:
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("zone points are normalised: 0 <= x, y <= 1")
        return v

    @field_validator("line")
    @classmethod
    def _line(cls, v):
        if v is None:
            return v
        for x, y in v:
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("line points are normalised: 0 <= x, y <= 1")
        if v[0] == v[1]:
            raise ValueError("a line needs two distinct points")
        return v

    @model_validator(mode="after")
    def _needs(self):
        if self.type == "zone_dwell" and not self.zone:
            raise ValueError("zone_dwell needs a zone")
        if self.type == "line_cross" and not self.line:
            raise ValueError("line_cross needs a line")
        return self


class TriggeredModel(BaseModel):
    model: str
    every_s: Optional[float] = Field(None, gt=0, le=3600)
    on_rule: Optional[str] = None
    classes: Optional[List[str]] = None
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    cooldown_s: float = Field(30.0, ge=0.0, le=86400)

    @model_validator(mode="after")
    def _when(self):
        if (self.every_s is None) == (self.on_rule is None):
            raise ValueError("set exactly one of every_s or on_rule")
        return self


class Shoplifting(BaseModel):
    """Per-person shoplifting detection; defaults are the standalone shoplifting app's."""
    model: str = "shoplifting_seq"
    # Samples per second fed into a person's window (120 samples: 40 s at 3 fps, as trained).
    sample_fps: float = Field(3.0, ge=0.5, le=10)
    # Smoothed Shoplifting probability that raises an alert.
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    smooth_window: int = Field(3, ge=1, le=30)
    # Real samples a window needs before it is judged.
    min_window_samples: int = Field(30, ge=1, le=120)
    # People shorter than this (box height, pixels) are too small to read a pose from.
    min_person_height_px: float = Field(60.0, ge=0, le=4000)
    min_confidence: float = Field(0.4, ge=0.0, le=1.0)
    track_cooldown_s: float = Field(300.0, ge=0, le=86400)
    camera_cooldown_s: float = Field(30.0, ge=0, le=86400)


class Tiling(BaseModel):
    mode: Literal["off", "auto", "on"] = "auto"
    grid: str = "2x2"  # columns x rows
    overlap: float = Field(0.2, ge=0.0, le=0.5)
    min_side: int = Field(960, ge=0, le=16384)
    # Boxes from different windows overlapping at least this much are one object.
    merge_iou: float = Field(0.5, ge=0.05, le=0.95)

    @field_validator("grid")
    @classmethod
    def _grid(cls, v: str) -> str:
        try:
            cols, rows = parse_grid(v)
        except ValueError:
            raise ValueError(f"grid must look like '2x2', not {v!r}") from None
        if cols < 1 or rows < 1 or 1 + cols * rows > MAX_TILE_WINDOWS:
            raise ValueError(f"grid {cols}x{rows} needs {1 + cols * rows} windows; at most {MAX_TILE_WINDOWS}")
        return f"{cols}x{rows}"

    def tile_config(self) -> TileConfig:
        return TileConfig(mode=self.mode, grid=parse_grid(self.grid), overlap=self.overlap,
                          min_side=self.min_side, merge_iou=self.merge_iou)


class CameraConfig(BaseModel):
    rules: List[Rule] = Field(default_factory=list)
    triggered: List[TriggeredModel] = Field(default_factory=list)
    shoplifting: Optional[Shoplifting] = None
    # Detections below this never reach the tracker.
    detect_confidence: float = Field(0.25, ge=0.05, le=1.0)
    # Restrict the primary detector to these class names (None = all).
    classes: Optional[List[str]] = None
    tiling: Tiling = Field(default_factory=Tiling)

    @model_validator(mode="after")
    def _refs(self):
        names = [r.name for r in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("rule names must be unique")
        for t in self.triggered:
            if t.on_rule and t.on_rule not in names:
                raise ValueError(f"triggered model refers to unknown rule '{t.on_rule}'")
        return self

    def models_used(self) -> List[str]:
        used = [t.model for t in self.triggered]
        used += [r.classify.model for r in self.rules if r.classify]
        if self.shoplifting is not None:
            used.append(self.shoplifting.model)
        return sorted(set(used))
