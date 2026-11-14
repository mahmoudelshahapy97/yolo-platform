"""End-to-end tests against a running stack, through nginx (HTTPS).

Run by tools/e2e.sh on a scratch stack (``-p yp-test``) whose model repository is a
copy, because the rollout tests add versions to it. Needs: E2E_BASE (https://nginx),
ADMIN_API_KEY, E2E_MODELS (the stack's model repository, mounted read-only),
E2E_VIDEOS (the clips).
"""

import json
import os
import time
import uuid
from pathlib import Path

import httpx
import pytest

BASE = os.environ.get("E2E_BASE", "https://nginx")
ADMIN = os.environ["ADMIN_API_KEY"]
MODELS = Path(os.environ.get("E2E_MODELS", "/models"))
VIDEOS = Path(os.environ.get("E2E_VIDEOS", "/videos"))
DATA = Path(__file__).resolve().parents[1] / "data"


def client(key=ADMIN, **kw):
    headers = {"X-API-Key": key} if key else {}
    return httpx.Client(base_url=BASE, headers=headers, verify=False, timeout=60, **kw)


@pytest.fixture(scope="session")
def admin():
    with client() as c:
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                if c.get("/health/ready").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(3)
        yield c


def img(name):
    return (name, (DATA / name).read_bytes(), "image/jpeg")


def wait_for(fn, timeout, every=2.0, what="condition"):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = fn()
        if last:
            return last
        time.sleep(every)
    raise AssertionError(f"timed out waiting for {what}; last={last!r}")


# ---------------- health, auth, tenancy ----------------

def test_ready_reports_all_models(admin):
    r = admin.get("/health/ready").json()
    assert r["status"] == "ready"
    assert set(r["models"]) == {"yolo26n_detection", "people_detection", "fire_detection", "ppe_detection", "yolo26n_cls",
                                "gender_detection", "gender0_detection", "yolo26l_detection", "lost_found_detection", "yolo26m_pose",
                                "yolo26n_seg", "yolo11n_seg", "shoplifting_seq"}


def test_http_redirects_and_metrics_hidden():
    with httpx.Client(verify=False) as c:
        r = c.get(BASE.replace("https://", "http://") + "/v1/models", follow_redirects=False)
        assert r.status_code == 301 and r.headers["location"].startswith("https://")
    with client() as c:
        assert c.get("/metrics").status_code == 404


def test_auth_required_and_rejected():
    with client(None) as c:
        assert c.get("/v1/models").status_code == 401
    with client("yp_deadbeef_notarealsecret") as c:
        assert c.get("/v1/models").status_code == 401


@pytest.fixture(scope="session")
def tenant_key(admin):
    name = f"acme-{uuid.uuid4().hex[:6]}"
    assert admin.post("/v1/admin/tenants", json={"name": name, "model_allowlist": ["yolo26n_detection"],
                                                  "max_cameras": 1}).status_code == 201
    r = admin.post("/v1/admin/keys", json={"tenant": name, "name": "e2e", "scopes": ["infer"], "rate_per_min": 6})
    assert r.status_code == 201
    return r.json()


def test_scope_allowlist_and_rate_limit(tenant_key):
    with client(tenant_key["api_key"]) as c:
        assert c.get("/v1/cameras").status_code == 403                      # no cameras scope
        r = c.post("/v1/infer/images", files={"file": img("people.jpg")}, data={"model": "fire_detection"})
        assert r.status_code == 403                                          # not on the allowlist
        models = c.get("/v1/models").json()["models"]
        assert [m["name"] for m in models] == ["yolo26n_detection"]
        codes = [c.post("/v1/infer/images", files={"file": img("people.jpg")}).status_code for _ in range(6)]
        assert codes[-1] == 429 and 200 in codes
        r = c.post("/v1/infer/images", files={"file": img("people.jpg")})
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_revoked_key_is_rejected(admin, tenant_key):
    assert admin.delete(f"/v1/admin/keys/{tenant_key['id']}").status_code == 200
    # Other API processes may cache the key for up to 30 s.
    with client(tenant_key["api_key"]) as c:
        wait_for(lambda: c.get("/v1/models").status_code == 401, 45, what="revocation")


# ---------------- inference ----------------

@pytest.mark.parametrize("model,image", [("yolo26n_detection", "street.jpg"), ("people_detection", "people.jpg"),
                                         ("ppe_detection", "street.jpg"), ("fire_detection", "fire.jpg"),
                                         ("gender_detection", "people.jpg"), ("gender0_detection", "people.jpg"),
                                         ("yolo26l_detection", "street.jpg"), ("lost_found_detection", "street.jpg")])
def test_detectors(admin, model, image):
    r = admin.post("/v1/infer/images", files={"file": img(image)}, data={"model": model, "conf": "0.2"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == model and body["task"] == "detect" and body["model_version"] == "1"
    im = body["images"][0]
    assert im["detections"], f"{model} found nothing on {image}"
    names = json.loads((MODELS / model / "model_meta.json").read_text())["names"].values()
    for d in im["detections"]:
        assert d["class_name"] in names and 0.2 <= d["confidence"] <= 1
        b = d["bbox"]
        assert 0 <= b["x1"] < b["x2"] <= im["width"] and 0 <= b["y1"] < b["y2"] <= im["height"]
    t = body["timing"]
    assert t["total_ms"] >= t["inference_ms"] > 0


def test_fire_model_labels_fire(admin):
    r = admin.post("/v1/infer/images", files={"file": img("fire.jpg")}, data={"model": "fire_detection", "conf": "0.2"})
    assert {d["class_name"] for d in r.json()["images"][0]["detections"]} <= {"fire", "smoke", "class_2"}


def test_classifier_and_crop_classification(admin):
    r = admin.post("/v1/infer/images", files={"file": img("street.jpg")}, data={"model": "yolo26n_cls", "top_k": "3"})
    assert r.status_code == 200
    cls = r.json()["images"][0]["classifications"]
    assert len(cls) == 3 and cls[0]["confidence"] >= cls[1]["confidence"] >= cls[2]["confidence"]
    r = admin.post("/v1/infer/images", files={"file": img("street.jpg")},
                   data={"classify_model": "yolo26n_cls", "classify_classes": "0", "top_k": "2"})
    dets = r.json()["images"][0]["detections"]
    people = [d for d in dets if d["class_id"] == 0]
    assert people and all(len(d["classification"]) == 2 for d in people)
    assert all("classification" not in d for d in dets if d["class_id"] != 0)


def test_batch_and_limits(admin):
    r = admin.post("/v1/infer/batch", files=[("files", img(n)) for n in ("people.jpg", "street.jpg", "fire.jpg")],
                   data={"model": "people_detection"})
    assert r.status_code == 200 and [i["filename"] for i in r.json()["images"]] == ["people.jpg", "street.jpg", "fire.jpg"]
    too_many = [("files", img("people.jpg"))] * 33
    assert admin.post("/v1/infer/batch", files=too_many).status_code == 413
    junk = ("x.jpg", b"not an image at all", "image/jpeg")
    assert admin.post("/v1/infer/images", files={"file": junk}).status_code == 422
    big = ("big.jpg", os.urandom(11 * 1024 * 1024), "image/jpeg")
    assert admin.post("/v1/infer/images", files={"file": big}).status_code == 413
    assert admin.post("/v1/infer/images", files={"file": img("people.jpg")}, data={"model": "nope"}).status_code == 404


def test_pose_returns_keypoints(admin):
    r = admin.post("/v1/infer/images", files={"file": img("people.jpg")}, data={"model": "yolo26m_pose"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "pose"
    im = body["images"][0]
    assert im["detections"]
    for d in im["detections"]:
        assert d["class_name"] == "person" and len(d["keypoints"]) == 17
        for x, y, c in d["keypoints"]:
            assert 0 <= x <= im["width"] and 0 <= y <= im["height"] and 0 <= c <= 1
    # Most confident person: most keypoints inside their box.
    d = im["detections"][0]
    b = d["bbox"]
    inside = [b["x1"] - 5 <= x <= b["x2"] + 5 and b["y1"] - 5 <= y <= b["y2"] + 5 for x, y, c in d["keypoints"] if c > 0.5]
    assert sum(inside) >= 0.8 * len(inside)


@pytest.mark.parametrize("model", ["yolo26n_seg", "yolo11n_seg"])
def test_segmentation_returns_polygons(admin, model):
    r = admin.post("/v1/infer/images", files={"file": img("street.jpg")}, data={"model": model})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "segment"
    dets = body["images"][0]["detections"]
    assert dets and sum(len(d["polygon"]) >= 3 for d in dets) >= 0.8 * len(dets)
    for d in dets:
        b = d["bbox"]
        for x, y in d["polygon"]:
            assert b["x1"] - 2 <= x <= b["x2"] + 2 and b["y1"] - 2 <= y <= b["y2"] + 2


def test_sequences_keypoints_and_features_agree(admin):
    import numpy as np
    from yolo_platform_shared.behavior import windows_from_sequence

    rng = np.random.default_rng(1)
    seqs = []
    for n in range(3):
        base = rng.uniform(0.3, 0.7, (17, 2))
        seqs.append([(base + rng.normal(0, 0.01, (17, 2))).tolist() if t % 7 else None for t in range(40 + 30 * n)])
    r = admin.post("/v1/infer/sequences", json={"model": "shoplifting_seq", "keypoints": seqs})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "sequence" and len(body["results"]) == 3
    for res in body["results"]:
        assert res["class_name"] in ("Normal", "Shoplifting")
        assert abs(sum(res["probabilities"].values()) - 1) < 1e-3
    feats = [windows_from_sequence([None if k is None else np.asarray(k, np.float32) for k in s]).tolist() for s in seqs]
    r2 = admin.post("/v1/infer/sequences", json={"features": feats})
    assert r2.status_code == 200, r2.text
    for a, b in zip(body["results"], r2.json()["results"]):
        assert abs(a["probabilities"]["Shoplifting"] - b["probabilities"]["Shoplifting"]) < 1e-3


def test_sequences_validation(admin):
    assert admin.post("/v1/infer/sequences", json={}).status_code == 422
    assert admin.post("/v1/infer/sequences", json={"keypoints": [[[[0.5, 0.5]] * 16]]}).status_code == 422
    assert admin.post("/v1/infer/sequences", json={"features": [[[[0.0] * 12] * 17] * 10]}).status_code == 422
    one = {"keypoints": [[[[0.5, 0.5]] * 17]]}
    assert admin.post("/v1/infer/sequences", json={**one, "model": "yolo26n_detection"}).status_code == 404
    r = admin.post("/v1/infer/images", files={"file": img("people.jpg")}, data={"model": "shoplifting_seq"})
    assert r.status_code == 422 and "sequences" in r.text


# ---------------- cameras ----------------

def test_camera_validation(admin):
    bad_rule = {"name": "x", "url": "rtsp://mediamtx:8554/sub0", "config": {"rules": [{"name": "z", "type": "zone_dwell"}]}}
    assert admin.post("/v1/cameras", json=bad_rule).status_code == 422
    bad_model = {"name": "x", "url": "rtsp://mediamtx:8554/sub0", "model": "yolo26n_cls"}
    assert admin.post("/v1/cameras", json=bad_model).status_code == 422
    unknown = {"name": "x", "url": "rtsp://mediamtx:8554/sub0", "config": {"triggered": [{"model": "nope", "every_s": 1}]}}
    assert admin.post("/v1/cameras", json=unknown).status_code == 422
    assert admin.post("/v1/cameras", json={"name": "x", "url": "ftp://cam"}).status_code == 422
    seg = {"name": "x", "url": "rtsp://mediamtx:8554/sub0", "model": "yolo26n_seg"}
    assert admin.post("/v1/cameras", json=seg).status_code == 422
    shop_on_detector = {"name": "x", "url": "rtsp://mediamtx:8554/sub0", "config": {"shoplifting": {}}}
    assert admin.post("/v1/cameras", json=shop_on_detector).status_code == 422
    shop_bad_model = {"name": "x", "url": "rtsp://mediamtx:8554/sub0", "model": "yolo26m_pose",
                      "config": {"shoplifting": {"model": "yolo26n_cls"}}}
    assert admin.post("/v1/cameras", json=shop_bad_model).status_code == 422


def test_pose_camera_raises_shoplifting_events(admin):
    # threshold 0 and a 3-sample window: every judgement alerts, so the whole path
    # (pose shm output, per-person windows, sequence model, event) shows up quickly.
    body = {"name": f"e2e-shop-{uuid.uuid4().hex[:6]}", "url": "rtsp://mediamtx:8554/sub0", "target_fps": 5,
            "model": "yolo26m_pose",
            "config": {"rules": [{"name": "people", "type": "presence", "cooldown_s": 1}],
                       "shoplifting": {"threshold": 0.0, "min_window_samples": 3, "sample_fps": 3,
                                       "camera_cooldown_s": 1, "track_cooldown_s": 1}}}
    cam = admin.post("/v1/cameras", json=body)
    assert cam.status_code == 201, cam.text
    cid = cam.json()["id"]
    try:
        assert admin.post(f"/v1/cameras/{cid}/start").json()["enabled"] is True
        events = wait_for(lambda: [e for e in admin.get(f"/v1/cameras/{cid}/events").json()
                                   if e["type"] == "shoplifting"], 120, what="shoplifting events")
        e = events[0]
        assert e["rule"] == "shoplifting" and e["track_id"] is not None and e["bbox"]
        assert 0 <= e["data"]["probability"] <= 1 and e["data"]["model"] == "shoplifting_seq"
        assert e["data"]["window_samples"] >= 3
        # Status is published every 2 s and events can beat the first "streaming" one.
        wait_for(lambda: (lambda s: s if s and s["state"] == "streaming" and s["fps"] >= 3.0 else None)(
            admin.get(f"/v1/cameras/{cid}").json()["status"]), 30, what="pose camera streaming")
        # The rules still run on the pose model's boxes.
        assert any(x["rule"] == "people" for x in admin.get(f"/v1/cameras/{cid}/events").json())
    finally:
        assert admin.delete(f"/v1/cameras/{cid}").status_code == 204


def test_camera_lifecycle_events_and_websocket(admin):
    body = {"name": f"e2e-{uuid.uuid4().hex[:6]}", "url": "rtsp://mediamtx:8554/sub0", "target_fps": 5,
            "config": {"rules": [{"name": "people", "type": "presence", "cooldown_s": 1},
                                 {"name": "many", "type": "count_above", "count": 1, "cooldown_s": 2}],
                       "triggered": [{"model": "ppe_detection", "on_rule": "people", "min_confidence": 0.3}]}}
    cam = admin.post("/v1/cameras", json=body)
    assert cam.status_code == 201, cam.text
    cid = cam.json()["id"]
    assert admin.post("/v1/cameras", json=body).status_code == 409                       # duplicate name
    try:
        assert admin.post(f"/v1/cameras/{cid}/start").json()["enabled"] is True

        def streaming():
            s = admin.get(f"/v1/cameras/{cid}").json()["status"]
            return s if s and s["state"] == "streaming" and s["fps"] >= 4.0 else None

        s = wait_for(streaming, 90, what="camera streaming at ~5 fps")
        assert s["backend"] in ("nvdec", "cpu") and s["owner"]
        events = wait_for(lambda: admin.get(f"/v1/cameras/{cid}/events").json(), 90, what="events")
        assert {e["rule"] for e in events} & {"people", "many"}
        e = events[0]
        assert e["camera_id"] == cid and e["ts"]

        import asyncio
        from websockets.asyncio.client import connect
        import ssl

        async def ws_messages():
            ctx = ssl.create_default_context()
            ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
            url = BASE.replace("https://", "wss://") + f"/v1/cameras/{cid}/stream?api_key={ADMIN}"
            kinds = set()
            async with connect(url, ssl=ctx) as ws:
                deadline = time.time() + 40
                while time.time() < deadline and kinds != {"status", "event"}:
                    kinds.add(json.loads(await asyncio.wait_for(ws.recv(), 30))["type"])
            return kinds

        assert asyncio.run(ws_messages()) == {"status", "event"}

        # Config change without a restart: rules hot-swap.
        r = admin.patch(f"/v1/cameras/{cid}", json={"config": {"rules": [{"name": "renamed", "type": "presence",
                                                                            "cooldown_s": 1}]}})
        assert r.status_code == 200
        wait_for(lambda: any(e["rule"] == "renamed" for e in admin.get(f"/v1/cameras/{cid}/events").json()), 90,
                 what="event from the updated rule")

        assert admin.post(f"/v1/cameras/{cid}/stop").json()["enabled"] is False
        wait_for(lambda: admin.get(f"/v1/cameras/{cid}").json()["status"] is None, 30, what="status cleared")
    finally:
        assert admin.delete(f"/v1/cameras/{cid}").status_code == 204


# ---------------- video jobs ----------------

def submit(admin, path: Path, **data):
    with path.open("rb") as f:
        r = admin.post("/v1/infer/videos", files={"file": (path.name, f, "video/mp4")},
                       data={k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in data.items()})
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


def job(admin, jid):
    return admin.get(f"/v1/jobs/{jid}").json()


def test_video_job_completes_with_results_and_annotation(admin):
    jid = submit(admin, VIDEOS / "sub" / "clip3.mp4", sample_fps=5, annotate=True)
    j = wait_for(lambda: (lambda x: x if x["status"] in ("completed", "failed") else None)(job(admin, jid)), 300,
                 what="job done")
    assert j["status"] == "completed", j
    assert j["progress"] == 100.0 and j["processed_frames"] == j["total_frames"] > 0
    assert j["summary"]["frames_processed"] == j["processed_frames"]
    for url in (j["result_url"], j["output_url"]):
        assert url.startswith("/storage/")
    lines = admin.get(j["result_url"]).text.strip().splitlines()
    assert len(lines) == j["processed_frames"]
    first = json.loads(lines[0])
    assert {"frame", "time_s", "detections"} <= set(first)
    video = admin.get(j["output_url"])
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4" and len(video.content) > 10_000


@pytest.mark.parametrize("model,field", [("yolo26m_pose", "keypoints"), ("yolo26n_seg", "polygon")])
def test_video_job_pose_and_segment(admin, model, field):
    # clip0 (cctv_video1) has people; clip3 is an office scene with none.
    jid = submit(admin, VIDEOS / "sub" / "clip0.mp4", model=model, sample_fps=2, annotate=True)
    j = wait_for(lambda: (lambda x: x if x["status"] in ("completed", "failed") else None)(job(admin, jid)), 300,
                 what="job done")
    assert j["status"] == "completed", j
    assert j["summary"]["task"] == ("pose" if field == "keypoints" else "segment")
    dets = [d for line in admin.get(j["result_url"]).text.strip().splitlines()
            for d in json.loads(line)["detections"]]
    assert dets and all(field in d for d in dets)
    assert admin.get(j["output_url"]).status_code == 200


def test_video_job_cancel(admin):
    jid = submit(admin, VIDEOS / "main" / "clip0.mp4", sample_fps=25)
    wait_for(lambda: job(admin, jid)["status"] == "processing", 120, every=0.5, what="processing")
    assert admin.post(f"/v1/jobs/{jid}/cancel").json()["status"] == "cancelled"
    time.sleep(5)
    assert job(admin, jid)["status"] == "cancelled"


def test_corrupt_video_is_retried_then_dead_lettered(admin):
    path = Path("/tmp/corrupt.mp4")
    path.write_bytes(os.urandom(200_000))
    jid = submit(admin, path)
    j = wait_for(lambda: (lambda x: x if x["status"] == "failed" else None)(job(admin, jid)), 240, what="job failed")
    assert j["attempts"] == 3 and "readable" in (j["error"] or "")


def test_video_rejects_unknown_model_and_type(admin):
    with (VIDEOS / "sub" / "clip3.mp4").open("rb") as f:
        assert admin.post("/v1/infer/videos", files={"file": ("a.mp4", f, "video/mp4")},
                          data={"model": "yolo26n_cls"}).status_code == 422
        f.seek(0)
        assert admin.post("/v1/infer/videos", files={"file": ("a.mp4", f, "video/mp4")},
                          data={"model": "shoplifting_seq"}).status_code == 422
    assert admin.post("/v1/infer/videos", files={"file": ("a.txt", b"x", "text/plain")}).status_code == 415


# ---------------- model rollout ----------------

def upload(admin, name, version, plan: Path, golden: Path):
    with plan.open("rb") as p, golden.open("rb") as g:
        return admin.post(f"/v1/models/{name}/versions/{version}",
                          files={"plan": ("model.plan", p), "golden": ("golden.npz", g)}, timeout=300)


def test_rollout_gate_and_rollback(admin):
    name = "yolo26n_detection"
    v1 = MODELS / name / "1"
    r = upload(admin, name, 2, v1 / "model.plan", v1 / "golden.npz")
    assert r.status_code == 201, r.text
    assert upload(admin, name, 2, v1 / "model.plan", v1 / "golden.npz").status_code == 409   # immutable

    r = admin.post(f"/v1/models/{name}/versions/2/deploy", timeout=300)
    assert r.status_code == 200, r.text
    assert r.json()["gate"]["passed"] and r.json()["previous"] == 1
    info = admin.get(f"/v1/models/{name}").json()
    assert info["active_version"] == 2
    assert wait_for(lambda: admin.post("/v1/infer/images", files={"file": img("street.jpg")}).json()["model_version"] == "2",
                    30, what="traffic on v2")

    r = admin.post(f"/v1/models/{name}/rollback", timeout=300)
    assert r.status_code == 200 and r.json()["version"] == 1
    assert admin.get(f"/v1/models/{name}").json()["active_version"] == 1

    # A version whose engine disagrees with its golden reference never goes live.
    bad_golden = MODELS / "fire_detection" / "1" / "golden.npz"
    assert upload(admin, name, 3, v1 / "model.plan", bad_golden).status_code == 201
    r = admin.post(f"/v1/models/{name}/versions/3/deploy", timeout=300)
    assert r.status_code == 422 and r.json()["detail"]["gate"]["passed"] is False
    versions = {v["version"]: v["status"] for v in admin.get(f"/v1/models/{name}").json()["versions"]}
    assert versions == {1: "active", 2: "retired", 3: "failed"}
    assert admin.post("/v1/infer/images", files={"file": img("street.jpg")}).json()["model_version"] == "1"


def test_deploy_requires_admin(admin):
    r = admin.post("/v1/admin/keys", json={"tenant": "default", "name": "infer-only", "scopes": ["infer"]})
    with client(r.json()["api_key"]) as c:
        assert c.post("/v1/models/yolo26n_detection/versions/1/deploy").status_code == 403


def _clip4_frame_jpeg():
    """A frame of the high, wide crowd view (main/clip4.mp4, 1920x1080)."""
    import cv2

    cap = cv2.VideoCapture(str(VIDEOS / "main" / "clip4.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 170)
    ok, frame = cap.read()
    cap.release()
    assert ok, "main/clip4.mp4 is missing (tools/gen_substreams.sh)"
    return ("crowd.jpg", cv2.imencode(".jpg", frame)[1].tobytes(), "image/jpeg")


def test_image_tiling_finds_the_crowd(admin):
    def people(tile):
        r = admin.post("/v1/infer/images", files={"file": _clip4_frame_jpeg()},
                       data={"model": "lost_found_detection", "conf": "0.25", "tile": tile})
        assert r.status_code == 200, r.text
        return sum(d["class_name"] == "person" for d in r.json()["images"][0]["detections"])

    plain, tiled = people("off"), people("on")
    assert tiled >= 10 and tiled >= 3 * max(plain, 1), (plain, tiled)
    r = admin.post("/v1/infer/images", files={"file": img("street.jpg")},
                   data={"model": "yolo26n_seg", "tile": "on"})
    assert r.status_code == 422


def test_tiled_camera_on_a_1080p_stream(admin):
    # Untiled, lost_found_detection sees ~2 people per frame here; tiled (auto, the
    # default, on a 1080p stream) ~20. The rule only fires if tiles reach the tracker.
    body = {"name": f"e2e-tiled-{uuid.uuid4().hex[:6]}", "url": "rtsp://mediamtx:8554/main4", "target_fps": 5,
            "model": "lost_found_detection",
            "config": {"rules": [{"name": "crowd", "type": "count_above", "count": 10, "classes": ["person"],
                                  "cooldown_s": 2}]}}
    cam = admin.post("/v1/cameras", json=body)
    assert cam.status_code == 201, cam.text
    cid = cam.json()["id"]
    try:
        assert cam.json()["config"]["tiling"]["mode"] == "auto"
        assert admin.post(f"/v1/cameras/{cid}/start").json()["enabled"] is True

        def streaming():
            s = admin.get(f"/v1/cameras/{cid}").json()["status"]
            return s if s and s["state"] == "streaming" and s["fps"] >= 4.0 else None

        wait_for(streaming, 120, what="tiled 1080p camera streaming at ~5 fps")
        events = wait_for(lambda: [e for e in admin.get(f"/v1/cameras/{cid}/events").json() if e["rule"] == "crowd"],
                          90, what="crowd event from tiled detections")
        assert events[0]["camera_id"] == cid
        bad = admin.patch(f"/v1/cameras/{cid}", json={"config": {"tiling": {"grid": "4x4"}}})
        assert bad.status_code == 422
    finally:
        assert admin.delete(f"/v1/cameras/{cid}").status_code == 204
