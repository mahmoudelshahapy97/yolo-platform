"""Per-key token bucket in Redis (one Lua call per request, atomic across API processes)."""

import time

from redis.asyncio import Redis

# KEYS[1] bucket; ARGV: capacity, refill per second, now (s), cost. Returns {allowed, tokens_left, retry_after_ms}.
_LUA = """
local cap = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local b = redis.call('HMGET', KEYS[1], 't', 'ts')
local tokens = tonumber(b[1]) or cap
local ts = tonumber(b[2]) or now
tokens = math.min(cap, tokens + (now - ts) * rate)
local allowed = 0
local retry = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry = math.ceil((cost - tokens) / rate * 1000)
end
redis.call('HSET', KEYS[1], 't', tokens, 'ts', now)
redis.call('PEXPIRE', KEYS[1], math.ceil(cap / rate * 1000) + 1000)
return {allowed, math.floor(tokens), retry}
"""


class RateLimiter:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis
        self._script = redis.register_script(_LUA)

    async def take(self, key_id: str, per_minute: int, cost: int = 1) -> tuple[bool, int, float]:
        """-> (allowed, tokens left, retry-after seconds). Burst capacity = one minute's allowance."""
        allowed, left, retry_ms = await self._script(
            keys=[f"rl:{key_id}"], args=[per_minute, per_minute / 60.0, time.time(), cost])
        return bool(allowed), int(left), retry_ms / 1000.0
