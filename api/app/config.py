from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    database_url: str = "postgresql://inference:inference@postgres:5432/inference"
    redis_url: str = "redis://redis:6379/0"
    triton_url: str = "triton:8001"
    triton_http_url: str = "http://triton:8000"
    model_repository: Path = Path("/models")

    # Bootstrap admin key, created (or rotated) on start. Format yp_<prefix>_<secret>.
    admin_api_key: str = ""

    # Request limits.
    max_image_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 8192 * 8192
    max_batch_images: int = 32
    max_video_bytes: int = 2 * 1024 * 1024 * 1024
    # Admission control: requests in flight per API process before answering 503.
    max_inflight: int = 64
    # Job backlog (queued video jobs) before new submissions get 503.
    max_job_backlog: int = 200

    # Object storage (MinIO or S3).
    s3_endpoint: str = "http://minio:9000"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = "yolo-platform"
    s3_region: str = "us-east-1"

    triton_timeout_s: float = 10.0
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
