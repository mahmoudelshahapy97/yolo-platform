"""Prometheus metrics. The API runs several uvicorn workers, so prometheus_client
multiprocess mode is used when PROMETHEUS_MULTIPROC_DIR is set (it is in the image)."""

import os

from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest,
                               multiprocess)

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1, 2.5, 5, 10)

REQUESTS = Counter("inference_requests_total", "Inference requests", ["model", "endpoint", "status"])
ERRORS = Counter("inference_errors_total", "Failed inference requests", ["model", "endpoint", "reason"])
LATENCY = Histogram("inference_latency_seconds", "End-to-end inference request latency", ["model", "endpoint"],
                    buckets=LATENCY_BUCKETS)
STAGE = Histogram("inference_stage_seconds", "Per-stage inference time", ["model", "stage"], buckets=LATENCY_BUCKETS)
IMAGES = Counter("inference_images_total", "Images inferred", ["model"])
INFLIGHT = Gauge("inference_inflight_requests", "Inference requests in flight", multiprocess_mode="livesum")
REJECTED = Counter("inference_rejected_total", "Requests rejected by admission control or rate limiting", ["reason"])
HTTP = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "route", "status"],
                 buckets=LATENCY_BUCKETS)


def render() -> tuple[bytes, str]:
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST
