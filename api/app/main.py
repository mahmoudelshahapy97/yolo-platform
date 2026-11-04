import logging
import time
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, Request
from redis.asyncio import Redis

from . import metrics
from .config import get_settings
from .deps import AppState
from .routes import admin, cameras, health, images, jobs, models
from .services.auth import KeyStore
from .services.inference_service import InferenceService
from .services.model_registry import ModelRegistry
from .services.rate_limit import RateLimiter
from .services.storage_service import Storage
from .services.triton_client import TritonClient

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    pool = await asyncpg.create_pool(s.database_url, min_size=2, max_size=20)
    redis = Redis.from_url(s.redis_url, decode_responses=True)
    triton = TritonClient(s.triton_url, s.triton_timeout_s)
    registry = ModelRegistry(pool, redis, triton, s.model_repository)
    storage = Storage(s.s3_endpoint, s.s3_access_key, s.s3_secret_key, s.s3_bucket, s.s3_region)
    core = AppState(settings=s, pool=pool, redis=redis, triton=triton, registry=registry,
                    inference=InferenceService(triton, registry), keys=KeyStore(pool), limiter=RateLimiter(redis),
                    storage=storage)
    app.state.core = core
    if s.admin_api_key:
        await core.keys.ensure_admin(s.admin_api_key)
    # Every API process runs this; the registry serialises it with a Redis lock,
    # so the first process loads and gates the models and the others find them active.
    await registry.bootstrap()
    await cameras.resync_desired(core)
    try:
        await storage.ensure_bucket()
    except Exception as exc:  # storage is needed for video jobs only
        logger.warning("object storage not reachable: %s", exc)
    yield
    await triton.close()
    await redis.aclose()
    await pool.close()


app = FastAPI(
    title="YOLO Inference Platform",
    version="1.0.0",
    lifespan=lifespan,
    description="Multi-model YOLO inference on NVIDIA Triton + TensorRT. Authenticate with the X-API-Key header.",
)


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", "unmatched")
    metrics.HTTP.labels(request.method, path, str(response.status_code)).observe(time.perf_counter() - start)
    return response


for r in (health.router, models.router, images.router, jobs.router, cameras.router, admin.router):
    app.include_router(r)
