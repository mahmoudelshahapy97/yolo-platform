"""Tenants and API keys (admin scope)."""

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..deps import require, state
from ..schemas import KeyIn, TenantIn
from ..services.auth import SCOPES, Principal, new_key

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.get("/tenants")
async def list_tenants(request: Request, p: Principal = Depends(require("admin"))):
    rows = await state(request).pool.fetch("SELECT id, name, model_allowlist, max_cameras, created_at FROM tenant ORDER BY name")
    return {"tenants": [dict(r) | {"id": str(r["id"])} for r in rows]}


@router.post("/tenants", status_code=201)
async def create_tenant(body: TenantIn, request: Request, p: Principal = Depends(require("admin"))):
    row = await state(request).pool.fetchrow(
        """INSERT INTO tenant (name, model_allowlist, max_cameras) VALUES ($1, $2, $3)
           ON CONFLICT (name) DO NOTHING RETURNING id""", body.name, body.model_allowlist, body.max_cameras)
    if row is None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"tenant '{body.name}' exists")
    return {"id": str(row["id"]), "name": body.name}


@router.post("/keys", status_code=201)
async def create_key(body: KeyIn, request: Request, p: Principal = Depends(require("admin"))):
    bad = [s for s in body.scopes if s not in SCOPES]
    if bad:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"unknown scopes {bad}; valid: {list(SCOPES)}")
    core = state(request)
    tenant_id = await core.pool.fetchval("SELECT id FROM tenant WHERE name = $1", body.tenant)
    if tenant_id is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown tenant '{body.tenant}'")
    key, prefix, digest = new_key()
    key_id = await core.pool.fetchval(
        """INSERT INTO api_key (tenant_id, name, prefix, secret_sha256, scopes, rate_per_min)
           VALUES ($1, $2, $3, $4, $5, $6) RETURNING id""",
        tenant_id, body.name, prefix, digest, body.scopes, body.rate_per_min)
    return {"id": str(key_id), "api_key": key, "note": "shown once; store it now"}


@router.get("/keys")
async def list_keys(request: Request, p: Principal = Depends(require("admin"))):
    rows = await state(request).pool.fetch(
        """SELECT k.id, t.name AS tenant, k.name, k.prefix, k.scopes, k.rate_per_min, k.created_at, k.last_used_at,
                  k.revoked_at FROM api_key k JOIN tenant t ON t.id = k.tenant_id ORDER BY k.created_at""")
    return {"keys": [dict(r) | {"id": str(r["id"])} for r in rows]}


@router.delete("/keys/{key_id}")
async def revoke_key(key_id: str, request: Request, p: Principal = Depends(require("admin"))):
    core = state(request)
    done = await core.pool.fetchval(
        "UPDATE api_key SET revoked_at = now() WHERE id::text = $1 AND revoked_at IS NULL RETURNING id", key_id)
    if done is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such active key")
    core.keys.forget()
    return {"revoked": key_id}
