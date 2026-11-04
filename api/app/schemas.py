from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from yolo_platform_shared.camera_config import CameraConfig


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ClassScore(BaseModel):
    class_id: int
    class_name: str
    confidence: float


class DetectionOut(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    classification: Optional[List[ClassScore]] = None


class ImageOut(BaseModel):
    index: int
    filename: Optional[str] = None
    width: int
    height: int
    detections: List[DetectionOut] = Field(default_factory=list)
    classifications: List[ClassScore] = Field(default_factory=list)


class TimingOut(BaseModel):
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float


class InferResponse(BaseModel):
    request_id: str
    model: str
    model_version: str
    task: str
    classify_model: Optional[str] = None
    classify_model_version: Optional[str] = None
    timing: TimingOut
    images: List[ImageOut]


class CameraIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    url: str = Field(..., min_length=7, max_length=2048, pattern=r"^(rtsp|rtsps|rtmp|http|https|file)://")
    target_fps: float = Field(5.0, gt=0, le=30)
    model: str = "yolo26n_detection"
    config: CameraConfig = Field(default_factory=CameraConfig)
    enabled: bool = False


class CameraPatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=128)
    url: Optional[str] = Field(None, min_length=7, max_length=2048, pattern=r"^(rtsp|rtsps|rtmp|http|https|file)://")
    target_fps: Optional[float] = Field(None, gt=0, le=30)
    model: Optional[str] = None
    config: Optional[CameraConfig] = None


class CameraOut(BaseModel):
    id: str
    name: str
    url: str
    enabled: bool
    target_fps: float
    model: str
    config: CameraConfig
    created_at: datetime
    updated_at: datetime
    status: Optional[dict] = None


class EventOut(BaseModel):
    id: int
    camera_id: str
    ts: datetime
    type: str
    rule: str
    track_id: Optional[int]
    class_name: Optional[str]
    confidence: Optional[float]
    bbox: Optional[dict]
    data: dict


class JobOut(BaseModel):
    job_id: str
    status: str
    model: str
    params: dict
    progress: float
    processed_frames: int
    total_frames: Optional[int]
    attempts: int
    error: Optional[str]
    summary: Optional[dict]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    result_url: Optional[str] = None
    output_url: Optional[str] = None


class TenantIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[\w\-.]+$")
    model_allowlist: Optional[List[str]] = None
    max_cameras: int = Field(100, ge=0, le=10000)


class KeyIn(BaseModel):
    tenant: str
    name: str = Field(..., min_length=1, max_length=64)
    scopes: List[str] = Field(default_factory=lambda: ["infer"])
    rate_per_min: int = Field(600, ge=1, le=1_000_000)
