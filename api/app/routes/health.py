from fastapi import APIRouter, Request, Response

from .. import metrics
from ..deps import state

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def live():
    return {"status": "alive"}


@router.get("/health/ready")
async def ready(request: Request, response: Response):
    core = state(request)
    checks = {}
    try:
        await core.pool.fetchval("SELECT 1")
        checks["postgres"] = True
    except Exception:
        checks["postgres"] = False
    try:
        checks["redis"] = bool(await core.redis.ping())
    except Exception:
        checks["redis"] = False
    checks["triton"] = await core.triton.ready()
    active = await core.registry.all_active() if checks["postgres"] else {}
    models = {}
    for name, m in active.items():
        models[name] = {"version": m.version, "triton_model": m.serve_name,
                        "ready": await core.triton.model_ready(m.serve_name)}
    checks["models"] = bool(models) and all(v["ready"] for v in models.values())
    ok = all(checks.values())
    response.status_code = 200 if ok else 503
    return {"status": "ready" if ok else "not_ready", "checks": checks, "models": models}


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    body, ctype = metrics.render()
    return Response(body, media_type=ctype)
