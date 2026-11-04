"""Camera ownership across shards: Redis leases, fair share, no central scheduler.

Every shard runs ``LeaseScheduler.tick()`` every couple of seconds:
  - heartbeat into the ``shards:alive`` sorted set (score = expiry time);
  - renew the leases it holds (only if still the owner: a Lua compare-and-pexpire);
  - release cameras that were disabled or deleted, or that it holds above its
    fair share (one per tick, so rebalancing is gradual);
  - claim unowned cameras with SET NX PX, up to its fair share.

fair share = ceil(enabled cameras / live shards), capped by SHARD_MAX_CAMERAS. Shards
publish their load to ``shards:load``; a shard above the floor share also gives one
back while another live shard is below the floor (so 7 cameras on 3 shards settle
at 3/2/2, not 3/3/1).
A shard that dies stops renewing; its leases expire after ``lease_ms`` and the
survivors claim them. A camera can never have two owners, because a lease is
only taken with NX and only renewed by the holder.
"""

import json
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Set

from redis import Redis

DESIRED_KEY = "cameras:desired"
ALIVE_KEY = "shards:alive"
LOAD_KEY = "shards:load"
LEASE_PREFIX = "camera:lease:"

_RENEW = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0
"""
_RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


@dataclass
class Assignment:
    start: List[str]          # camera ids newly owned
    stop: List[str]           # camera ids no longer owned
    owned: Set[str]
    desired: Dict[str, dict]  # all enabled cameras (id -> config)


class LeaseScheduler:
    def __init__(self, redis: Redis, shard_id: str, max_cameras: int, lease_ms: int = 15000) -> None:
        self.r = redis
        self.shard_id = shard_id
        self.max_cameras = max_cameras
        self.lease_ms = lease_ms
        self.owned: Set[str] = set()
        self._renew = redis.register_script(_RENEW)
        self._release = redis.register_script(_RELEASE)

    def live_shards(self) -> List[str]:
        now = time.time()
        self.r.zremrangebyscore(ALIVE_KEY, "-inf", now)
        return self.r.zrangebyscore(ALIVE_KEY, now, "+inf") or [self.shard_id]

    def fair_share(self, total: int, shards: int) -> int:
        return min(self.max_cameras, math.ceil(total / shards)) if total else 0

    def tick(self) -> Assignment:
        now = time.time()
        self.r.zadd(ALIVE_KEY, {self.shard_id: now + self.lease_ms / 1000})
        desired = {k: json.loads(v) for k, v in self.r.hgetall(DESIRED_KEY).items()}
        stop: List[str] = []

        # Renew; anything we could not renew is no longer ours.
        for cam in list(self.owned):
            if cam not in desired:
                self._release(keys=[LEASE_PREFIX + cam], args=[self.shard_id])
                self.owned.discard(cam)
                stop.append(cam)
            elif not self._renew(keys=[LEASE_PREFIX + cam], args=[self.shard_id, self.lease_ms]):
                self.owned.discard(cam)
                stop.append(cam)

        live = self.live_shards()
        share = self.fair_share(len(desired), len(live))
        floor = min(self.max_cameras, len(desired) // len(live))
        loads = dict(zip(live, (int(v or 0) for v in self.r.hmget(LOAD_KEY, live))))
        loads[self.shard_id] = len(self.owned)
        starved = any(n < floor for sid, n in loads.items() if sid != self.shard_id)
        # Give back one camera per tick when above the fair share, or above the floor
        # while another shard is below it. No flapping: having released, this shard
        # will not claim the camera back while it is at its share; the starved shard
        # picks it up.
        if len(self.owned) > share or (starved and len(self.owned) > floor):
            cam = sorted(self.owned)[-1]
            self._release(keys=[LEASE_PREFIX + cam], args=[self.shard_id])
            self.owned.discard(cam)
            stop.append(cam)

        # While another shard is starved, claim only up to the floor, leaving it the rest.
        limit = floor if starved else share
        start: List[str] = []
        if len(self.owned) < limit:
            free = [c for c in desired if c not in self.owned]
            random.shuffle(free)  # spread claims when several shards start together
            for cam in free:
                if len(self.owned) >= limit:
                    break
                if self.r.set(LEASE_PREFIX + cam, self.shard_id, nx=True, px=self.lease_ms):
                    self.owned.add(cam)
                    start.append(cam)
        self.r.hset(LOAD_KEY, self.shard_id, len(self.owned))
        return Assignment(start=start, stop=stop, owned=set(self.owned), desired=desired)

    def release_all(self) -> None:
        for cam in list(self.owned):
            self._release(keys=[LEASE_PREFIX + cam], args=[self.shard_id])
        self.owned.clear()
        self.r.zrem(ALIVE_KEY, self.shard_id)
        self.r.hdel(LOAD_KEY, self.shard_id)
