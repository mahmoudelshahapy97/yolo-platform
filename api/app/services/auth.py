"""API keys: yp_<prefix>_<secret>. Only sha256(secret) is stored.

Scopes: infer, cameras, jobs, admin (admin implies all).
"""

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import asyncpg

SCOPES = ("infer", "cameras", "jobs", "admin")
_CACHE_TTL = 30.0


@dataclass(frozen=True)
class Principal:
    key_id: str
    tenant_id: str
    tenant_name: str
    scopes: Tuple[str, ...]
    model_allowlist: Optional[Tuple[str, ...]]
    rate_per_min: int

    def has(self, scope: str) -> bool:
        return "admin" in self.scopes or scope in self.scopes

    def may_use(self, model: str) -> bool:
        return self.model_allowlist is None or model in self.model_allowlist


def new_key() -> Tuple[str, str, str]:
    """-> (full key shown once, prefix, sha256 of secret)."""
    prefix = secrets.token_hex(4)
    secret = secrets.token_urlsafe(32)
    return f"yp_{prefix}_{secret}", prefix, hashlib.sha256(secret.encode()).hexdigest()


def split_key(key: str) -> Optional[Tuple[str, str]]:
    parts = key.split("_", 2)
    if len(parts) != 3 or parts[0] != "yp" or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


class KeyStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool
        self._cache: Dict[str, Tuple[float, Principal]] = {}

    async def verify(self, key: str) -> Optional[Principal]:
        split = split_key(key)
        if not split:
            return None
        prefix, secret = split
        digest = hashlib.sha256(secret.encode()).hexdigest()
        hit = self._cache.get(digest)
        if hit and time.monotonic() - hit[0] < _CACHE_TTL:
            return hit[1]
        row = await self.pool.fetchrow(
            """SELECT k.id, k.secret_sha256, k.scopes, k.rate_per_min, t.id AS tenant_id, t.name, t.model_allowlist
               FROM api_key k JOIN tenant t ON t.id = k.tenant_id
               WHERE k.prefix = $1 AND k.revoked_at IS NULL""", prefix)
        if row is None or not hmac.compare_digest(row["secret_sha256"], digest):
            return None
        principal = Principal(
            key_id=str(row["id"]), tenant_id=str(row["tenant_id"]), tenant_name=row["name"],
            scopes=tuple(row["scopes"]), rate_per_min=row["rate_per_min"],
            model_allowlist=tuple(row["model_allowlist"]) if row["model_allowlist"] is not None else None,
        )
        self._cache[digest] = (time.monotonic(), principal)
        await self.pool.execute("UPDATE api_key SET last_used_at = now() WHERE id = $1", row["id"])
        return principal

    def forget(self) -> None:
        self._cache.clear()

    async def ensure_admin(self, key: str) -> None:
        """Create the bootstrap tenant + admin key from ADMIN_API_KEY (idempotent; rotates the secret)."""
        split = split_key(key)
        if not split:
            raise ValueError("ADMIN_API_KEY must look like yp_<prefix>_<secret>")
        prefix, secret = split
        digest = hashlib.sha256(secret.encode()).hexdigest()
        async with self.pool.acquire() as conn, conn.transaction():
            tenant_id = await conn.fetchval(
                """INSERT INTO tenant (name) VALUES ('default')
                   ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id""")
            await conn.execute(
                """INSERT INTO api_key (tenant_id, name, prefix, secret_sha256, scopes, rate_per_min)
                   VALUES ($1, 'bootstrap admin', $2, $3, ARRAY['admin'], 100000)
                   ON CONFLICT (prefix) DO UPDATE SET secret_sha256 = EXCLUDED.secret_sha256, revoked_at = NULL""",
                tenant_id, prefix, digest)
