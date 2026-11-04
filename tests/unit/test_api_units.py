"""api/: key format, rate limiter (real Redis), Triton load config override. Runs in the API image."""

import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from app.services.auth import Principal, new_key, split_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def test_new_key_format_and_hash():
    key, prefix, digest = new_key()
    p, secret = split_key(key)
    assert key.startswith("yp_") and p == prefix and len(secret) >= 40
    assert digest == hashlib.sha256(secret.encode()).hexdigest()


@pytest.mark.parametrize("bad", ["", "yp_", "yp__x", "xx_abc_def", "yp_abc", "Bearer yp_a_b"])
def test_split_key_rejects(bad):
    assert split_key(bad) is None


def test_principal_scopes_and_allowlist():
    p = Principal("k", "t", "tenant", ("infer",), ("yolo26n_detection",), 10)
    assert p.has("infer") and not p.has("cameras")
    assert p.may_use("yolo26n_detection") and not p.may_use("ppe_detection")
    admin = Principal("k", "t", "tenant", ("admin",), None, 10)
    assert admin.has("cameras") and admin.may_use("anything")


def test_rate_limiter_token_bucket():
    url = os.environ.get("TEST_REDIS_URL")
    if not url:
        pytest.skip("TEST_REDIS_URL not set")
    from redis.asyncio import Redis

    from app.services.rate_limit import RateLimiter

    async def run():
        r = Redis.from_url(url, decode_responses=True)
        await r.delete("rl:test-key")
        rl = RateLimiter(r)
        results = [await rl.take("test-key", per_minute=5) for _ in range(6)]
        allowed = [a for a, _, _ in results]
        assert allowed == [True] * 5 + [False]
        assert results[-1][2] > 0                       # retry-after
        ok, _, _ = await rl.take("test-key", per_minute=5, cost=0)
        assert ok
        await r.delete("rl:test-key")
        await r.aclose()

    asyncio.run(run())


def test_triton_load_config_overrides_only_version_policy():
    from google.protobuf import json_format, text_format
    from tritonclient.grpc import model_config_pb2

    text = (ROOT / "inference/model_repository/yolo26n_detection/config.pbtxt").read_text()
    cfg = text_format.Parse(text, model_config_pb2.ModelConfig())
    cfg.version_policy.Clear()
    cfg.version_policy.specific.versions.extend([1, 2])
    d = json.loads(json.dumps(json_format.MessageToDict(cfg)))
    # int64 fields are strings in protobuf JSON; Triton parses them back.
    assert [int(v) for v in d["versionPolicy"]["specific"]["versions"]] == [1, 2]
    assert d["maxBatchSize"] == 16 and d["input"][0]["dataType"] == "TYPE_UINT8"
    assert d["dynamicBatching"]["preferredBatchSize"] == [4, 8, 16]
