"""workers/scheduler.py against a real Redis (REDIS_URL, a scratch DB). Skipped without one."""

import json
import os
import time
import uuid

import pytest
from redis import Redis

from workers import scheduler as sched
from workers.scheduler import LeaseScheduler


@pytest.fixture
def redis():
    url = os.environ.get("TEST_REDIS_URL")
    if not url:
        pytest.skip("TEST_REDIS_URL not set")
    r = Redis.from_url(url, decode_responses=True)
    r.flushdb()
    # Separate keys per test run.
    yield r
    r.flushdb()


def desire(r, n):
    r.delete(sched.DESIRED_KEY)
    if n:
        r.hset(sched.DESIRED_KEY, mapping={f"cam{i}": json.dumps({"id": f"cam{i}", "url": "x", "model": "m"})
                                           for i in range(n)})


def settle(*shards, ticks=12):
    for _ in range(ticks):
        for s in shards:
            s.tick()


def test_single_shard_takes_everything_up_to_its_cap(redis):
    desire(redis, 5)
    a = LeaseScheduler(redis, "a", max_cameras=3)
    settle(a)
    assert len(a.owned) == 3


def test_fair_share_and_no_double_ownership(redis):
    desire(redis, 7)
    a, b, c = (LeaseScheduler(redis, n, max_cameras=40) for n in "abc")
    settle(a, b, c)
    owned = [a.owned, b.owned, c.owned]
    assert sorted(len(o) for o in owned) == [2, 2, 3]
    assert set().union(*owned) == {f"cam{i}" for i in range(7)}
    assert sum(len(o) for o in owned) == 7          # disjoint
    for cam in a.owned:
        assert redis.get(sched.LEASE_PREFIX + cam) == "a"


def test_rebalances_when_a_shard_joins(redis):
    desire(redis, 6)
    a = LeaseScheduler(redis, "a", max_cameras=40)
    settle(a)
    assert len(a.owned) == 6
    b = LeaseScheduler(redis, "b", max_cameras=40)
    settle(a, b)
    assert (len(a.owned), len(b.owned)) == (3, 3)


def test_disabled_camera_is_released(redis):
    desire(redis, 2)
    a = LeaseScheduler(redis, "a", max_cameras=40)
    settle(a)
    redis.hdel(sched.DESIRED_KEY, "cam1")
    res = a.tick()
    assert "cam1" in res.stop and a.owned == {"cam0"}
    assert redis.get(sched.LEASE_PREFIX + "cam1") is None


def test_dead_shard_leases_expire_and_are_taken_over(redis):
    desire(redis, 4)
    a = LeaseScheduler(redis, "a", max_cameras=40, lease_ms=600)
    b = LeaseScheduler(redis, "b", max_cameras=40, lease_ms=600)
    settle(a, b)
    assert len(b.owned) == 2
    time.sleep(0.8)                                   # b stops ticking: dies without releasing
    settle(a, ticks=3)
    assert len(a.owned) == 4
    # A lease is renewed only by its holder: b coming back cannot keep what a took.
    lost = set(b.owned)
    res = b.tick()
    assert set(res.stop) >= lost and not (b.owned & a.owned)


def test_release_all(redis):
    desire(redis, 3)
    a = LeaseScheduler(redis, "a", max_cameras=40)
    settle(a)
    a.release_all()
    assert not any(redis.get(sched.LEASE_PREFIX + f"cam{i}") for i in range(3))
    assert redis.zscore(sched.ALIVE_KEY, "a") is None


@pytest.mark.parametrize("cams,shards", [(1, 3), (7, 3), (10, 4), (25, 4), (100, 4), (5, 5), (13, 2)])
def test_balanced_for_many_sizes(redis, cams, shards):
    desire(redis, cams)
    ss = [LeaseScheduler(redis, f"s{i}", max_cameras=40) for i in range(shards)]
    settle(*ss, ticks=20)
    loads = [len(s.owned) for s in ss]
    assert sum(loads) == cams and max(loads) - min(loads) <= 1, loads
    assert len(set().union(*(s.owned for s in ss))) == cams
