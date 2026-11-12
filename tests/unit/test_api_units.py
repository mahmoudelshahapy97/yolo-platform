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


def test_materialize_writes_a_per_version_serving_model(tmp_path):
    import shutil

    from google.protobuf import text_format
    from tritonclient.grpc import model_config_pb2

    from app.services.model_registry import ModelRegistry, serve_name

    repo, serving = tmp_path / "models", tmp_path / "serving"
    for name in ("yolo26n_detection", "yolo26n_cls"):
        (repo / name / "3").mkdir(parents=True)
        shutil.copy(ROOT / "inference/model_repository" / name / "config.pbtxt", repo / name / "config.pbtxt")
        shutil.copy(ROOT / "inference/model_repository" / name / "labels.txt", repo / name / "labels.txt")
        (repo / name / "3" / "model.plan").write_bytes(b"plan")
    serving.mkdir()
    reg = ModelRegistry(None, None, None, repo, serving, triton_models_path="/models")
    for name in ("yolo26n_detection", "yolo26n_cls"):
        sname = reg.materialize(name, 3)
        assert sname == serve_name(name, 3) == f"{name}__v3"
        cfg = text_format.Parse((serving / sname / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
        assert cfg.name == sname and not cfg.HasField("version_policy")
        assert cfg.input[0].data_type == model_config_pb2.TYPE_UINT8 and cfg.dynamic_batching.preferred_batch_size
        link = serving / sname / "1" / "model.plan"
        assert link.is_symlink() and str(link.readlink()) == f"/models/{name}/3/model.plan"
    # Re-materialising replaces the directory atomically.
    assert reg.materialize("yolo26n_cls", 3) == "yolo26n_cls__v3"
    assert sorted(p.name for p in serving.iterdir()) == ["yolo26n_cls__v3", "yolo26n_detection__v3"]


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


# ---------------- review fixes 2026-11-12 ----------------

# Route modules import app.metrics; the image's multiprocess dir is made by its entrypoint, not here.
if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
    os.makedirs(os.environ["PROMETHEUS_MULTIPROC_DIR"], exist_ok=True)

def _features(value):
    from yolo_platform_shared.behavior import NUM_FEATURES, NUM_FRAMES, NUM_KEYPOINTS
    w = [[[0.0] * NUM_FEATURES for _ in range(NUM_KEYPOINTS)] for _ in range(NUM_FRAMES)]
    w[5][3][2] = value
    return [w]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1e39])
def test_sequence_features_must_be_finite(bad):
    from fastapi import HTTPException
    from app.routes.sequences import _windows
    from app.schemas import SequenceIn
    with pytest.raises(HTTPException) as e:
        _windows(SequenceIn(features=_features(bad)))
    assert e.value.status_code == 422
    assert _windows(SequenceIn(features=_features(0.5))).shape == (1, 120, 17, 12)


class _Core:
    def __init__(self, inflight, max_inflight=4):
        from types import SimpleNamespace
        self.inflight = inflight
        self.settings = SimpleNamespace(max_inflight=max_inflight)
        self.seen_inflight = None
        core = self

        class Inference:
            async def sequences(self, windows, model):
                core.seen_inflight = core.inflight
                raise ValueError("stop here")
        self.inference = Inference()


def _request(core):
    from types import SimpleNamespace
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(core=core)))


def _principal():
    return Principal(key_id="k", tenant_id="t", tenant_name="n", scopes=("infer",), model_allowlist=None,
                     rate_per_min=60)


def test_sequences_respect_and_count_inflight():
    from fastapi import HTTPException
    from app.routes import sequences
    from app.schemas import SequenceIn
    body = SequenceIn(features=_features(0.5))
    full = _Core(inflight=4)
    with pytest.raises(HTTPException) as e:
        asyncio.run(sequences.infer_sequences(body, _request(full), _principal()))
    assert e.value.status_code == 503 and full.inflight == 4
    core = _Core(inflight=1)
    with pytest.raises(HTTPException) as e:
        asyncio.run(sequences.infer_sequences(body, _request(core), _principal()))
    assert e.value.status_code == 422          # the stub's ValueError
    assert core.seen_inflight == 2 and core.inflight == 1


def test_camera_rejects_pose_models_wider_than_the_shard_slot():
    from types import SimpleNamespace
    from fastapi import HTTPException
    from app.routes import cameras
    from app.services.model_registry import ActiveModel
    from yolo_platform_shared.camera_config import CameraConfig

    def core_with(kpt_shape):
        m = ActiveModel("p", 1, "pose", 640, 8, {0: "person"}, kpt_shape=kpt_shape)

        class Registry:
            async def active(self, name):
                return m
        return SimpleNamespace(registry=Registry())

    p = _principal()
    asyncio.run(cameras._check_models(core_with((17, 3)), p, "p", CameraConfig()))
    asyncio.run(cameras._check_models(core_with(None), p, "p", CameraConfig()))
    with pytest.raises(HTTPException) as e:
        asyncio.run(cameras._check_models(core_with((21, 3)), p, "p", CameraConfig()))
    assert e.value.status_code == 422 and "21 x 3" in e.value.detail


def test_registry_reads_model_meta_once_per_version(tmp_path):
    from app.services.model_registry import ModelRegistry
    (tmp_path / "m" / "1").mkdir(parents=True)
    (tmp_path / "m" / "1" / "model_meta.json").write_text(json.dumps({"imgsz": 640, "kpt_shape": [17, 3]}))
    reg = ModelRegistry.__new__(ModelRegistry)
    reg.repository = tmp_path
    reg._io = {}
    reads = []
    real = ModelRegistry._meta
    reg._meta = lambda n, v: reads.append((n, v)) or real(reg, n, v)

    async def go():
        a = await reg._io_fields("m", 1)
        b = await reg._io_fields("m", 1)
        missing = [await reg._io_fields("gone", 1) for _ in range(2)]
        return a, b, missing
    a, b, missing = asyncio.run(go())
    assert a == b and a["kpt_shape"] == [17, 3]
    assert reads.count(("m", 1)) == 1          # cached
    assert reads.count(("gone", 1)) == 2       # absent meta is not cached
    assert missing[0]["outputs"] == ["output0"]
