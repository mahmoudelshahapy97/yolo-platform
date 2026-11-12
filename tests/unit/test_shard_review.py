"""workers/camera_shard.py: group selection, slot width, and stopped cameras (no Triton, no Redis)."""

import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("tritonclient")

from workers import camera_shard as cs  # noqa: E402
from workers.rules import Event  # noqa: E402


def model(task="detect", imgsz=640, kpt_shape=None):
    return {"task": task, "imgsz": imgsz, "kpt_shape": kpt_shape, "max_batch": 8, "serve_name": "x", "names": {0: "p"}}


def test_servable_primary():
    assert cs.servable_primary(model())
    assert cs.servable_primary(model("pose", kpt_shape=(17, 3)))
    assert not cs.servable_primary(None)
    assert not cs.servable_primary(model("segment"))
    assert not cs.servable_primary(model(imgsz=320))
    assert not cs.servable_primary(model("pose", kpt_shape=(21, 3)))   # 69 columns > slot's 57


class _Source:
    def __init__(self):
        self.taken = 0

    def has_fresh(self):
        return True

    def take(self, buf):
        self.taken += 1
        return None          # nothing to infer: the loop just goes round again


def _shard(models, cams):
    sh = cs.Shard.__new__(cs.Shard)
    sh.s = SimpleNamespace(triton_url="unused")
    sh.models = SimpleNamespace(get=models.get)
    sh.cam_lock = threading.Lock()
    sh.cameras = {c.id: c for c in cams}
    sh.stop = threading.Event()
    sh._rr = 0
    sh._unservable_warned = {}
    return sh


def _cam(cam_id, model_name):
    return SimpleNamespace(id=cam_id, spec={"model": model_name}, model=model_name, source=_Source())


def test_unservable_majority_group_does_not_starve_valid_cameras(monkeypatch):
    bad = [_cam(f"bad{i}", "seg_model") for i in range(5)]
    good = [_cam(f"good{i}", "people") for i in range(3)]
    sh = _shard({"seg_model": model("segment"), "people": model()}, bad + good)
    slot = SimpleNamespace(inputs=[None] * cs.SLOT_BATCH, close=lambda: None)
    monkeypatch.setattr(cs.grpcclient, "InferenceServerClient", lambda url: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(sh, "_new_slot", lambda client, i: slot, raising=False)
    calls = {"n": 0}
    real_fresh = sh._fresh_by_model

    def fresh():
        calls["n"] += 1
        if calls["n"] > 20:
            sh.stop.set()
        return real_fresh()
    monkeypatch.setattr(sh, "_fresh_by_model", fresh, raising=False)
    sh._pipeline(0)
    assert all(c.source.taken > 0 for c in good)
    assert all(c.source.taken == 0 for c in bad)


def test_stopped_camera_emits_nothing_and_drops_shoplifting():
    sent = []
    sh = _shard({}, [])
    sh.s = SimpleNamespace(shard_id="s1")
    sh.redis = SimpleNamespace(xadd=lambda *a, **k: sent.append(a), delete=lambda *a: None)
    rt = SimpleNamespace(id="c1", spec={"tenant_id": "t"}, lock=threading.Lock(), stopped=False,
                         shoplifting=object(), source=SimpleNamespace(stop=lambda: None))
    sh.cameras = {"c1": rt}
    ev = Event(rule="r", type="shoplifting", ts=1.0, track_id=1, class_name="person", confidence=0.9, box=None, data={})
    sh._emit(rt, ev)
    assert len(sent) == 1
    sh._stop_camera("c1", "lease released")
    assert rt.stopped and rt.shoplifting is None
    sh._emit(rt, ev)
    assert len(sent) == 1
