"""Shared state (built in main.lifespan) and request dependencies: auth, scopes, rate limits."""

from dataclasses import dataclass
from typing import Optional

import asyncpg
from fastapi import Depends, HTTPException, Request, WebSocket, status
from fastapi.security import APIKeyHeader
from redis.asyncio import Redis

from .config import Settings
from .services.auth import KeyStore, Principal
from .services.inference_service import InferenceService
from .services.model_registry import ModelRegistry
from .services.rate_limit import RateLimiter
from .services.storage_service import Storage
from .services.triton_client import TritonClient


@dataclass
class AppState:
    settings: Settings
    pool: asyncpg.Pool
    redis: Redis
    triton: TritonClient
    registry: ModelRegistry
    inference: InferenceService
    keys: KeyStore
    limiter: RateLimiter
    storage: Storage
    inflight: int = 0


def state(request: Request) -> AppState:
    return request.app.state.core


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def principal(request: Request, key: Optional[str] = Depends(api_key_header)) -> Principal:
    if not key:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            key = auth[7:].strip()
    if not key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing API key (X-API-Key header)")
    p = await state(request).keys.verify(key)
    if p is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or revoked API key")
    request.state.principal = p
    return p


def require(scope: str, cost: int = 1):
    """Dependency: authenticated, has ``scope``, and within its rate limit."""

    async def dep(request: Request, p: Principal = Depends(principal)) -> Principal:
        if not p.has(scope):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"API key lacks the '{scope}' scope")
        allowed, left, retry = await state(request).limiter.take(p.key_id, p.rate_per_min, cost)
        if not allowed:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded",
                                headers={"Retry-After": str(max(1, int(retry + 0.999)))})
        return p

    return dep


async def ws_principal(ws: WebSocket, scope: str) -> Optional[Principal]:
    """WebSockets cannot always set headers from browsers: accept ?api_key= as well."""
    key = ws.headers.get("x-api-key") or ws.query_params.get("api_key")
    if not key:
        return None
    p = await ws.app.state.core.keys.verify(key)
    if p is None or not p.has(scope):
        return None
    return p


def check_model(p: Principal, model: str) -> None:
    if not p.may_use(model):
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"tenant may not use model '{model}'")
