"""Failure drills for tools/e2e.sh: kill a component mid-work, check the platform recovers.

    python tools/drills.py shard-failover --project yp-test
    python tools/drills.py job-reclaim --project yp-test
    python tools/drills.py triton-restart --project yp-test

Talks to the API through nginx and to Docker through /var/run/docker.sock (mounted
by e2e.sh), so it can kill containers the way a crash would (SIGKILL: no graceful
lease release, no job acknowledgement).
"""

import argparse
import os
import sys
import time

import httpx

BASE = os.environ.get("E2E_BASE", "https://nginx")
KEY = os.environ["ADMIN_API_KEY"]


def api():
    return httpx.Client(base_url=BASE, headers={"X-API-Key": KEY}, verify=False, timeout=60)


def docker():
    return httpx.Client(transport=httpx.HTTPTransport(uds="/var/run/docker.sock"), base_url="http://docker", timeout=60)


def containers(d, project, service):
    flt = f'{{"label":["com.docker.compose.project={project}","com.docker.compose.service={service}"],"status":["running"]}}'
    return d.get("/containers/json", params={"filters": flt}).raise_for_status().json()


def wait(fn, timeout, every=1.0, what=""):
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = fn()
        if v:
            return v
        time.sleep(every)
    raise SystemExit(f"FAIL: timed out waiting for {what}")


def shard_failover(project: str) -> None:
    a, d = api(), docker()
    shards = containers(d, project, "camera-shard")
    if len(shards) < 2:
        raise SystemExit("FAIL: need 2 running shards")
    ids = []
    for i in range(4):
        r = a.post("/v1/cameras", json={"name": f"drill-{i}", "url": f"rtsp://mediamtx:8554/sub{i}", "enabled": True,
                                        "config": {"rules": [{"name": "p", "type": "presence"}]}})
        r.raise_for_status()
        ids.append(r.json()["id"])
    try:
        def all_streaming(exclude=None):
            now = time.time()
            out = {}
            for cid in ids:
                s = a.get(f"/v1/cameras/{cid}").json()["status"]
                if not s or s["state"] != "streaming" or now - s["updated_at"] > 5 or s["fps"] < 3:
                    return None
                if exclude and s["owner"].startswith(exclude):
                    return None
                out[cid] = s["owner"]
            return out

        owners = wait(all_streaming, 120, what="4 cameras streaming")
        by_shard = {}
        for cid, owner in owners.items():
            by_shard.setdefault(owner, []).append(cid)
        print("before:", {k: len(v) for k, v in by_shard.items()})
        if len(by_shard) != 2:
            raise SystemExit(f"FAIL: expected the 4 cameras spread over 2 shards, got {by_shard}")
        victim_host = next(iter(by_shard)).rsplit("-", 1)[0]
        victim = next(c for c in shards if c["Id"].startswith(victim_host))
        t0 = time.time()
        d.post(f"/containers/{victim['Id']}/kill").raise_for_status()
        after = wait(lambda: all_streaming(exclude=victim_host), 90, what="cameras re-leased by the survivor")
        print(f"after {time.time() - t0:.0f}s: all 4 streaming on {set(after.values())}")
        d.post(f"/containers/{victim['Id']}/start").raise_for_status()
    finally:
        for cid in ids:
            a.delete(f"/v1/cameras/{cid}")
    print("shard-failover OK")


def job_reclaim(project: str) -> None:
    a, d = api(), docker()
    with open("/videos/main/clip0.mp4", "rb") as f:
        r = a.post("/v1/infer/videos", files={"file": ("clip0.mp4", f, "video/mp4")}, data={"sample_fps": "25"})
    r.raise_for_status()
    jid = r.json()["job_id"]
    wait(lambda: a.get(f"/v1/jobs/{jid}").json()["processed_frames"] > 50, 120, every=0.5, what="job progressing")
    worker = containers(d, project, "video-worker")[0]
    d.post(f"/containers/{worker['Id']}/kill").raise_for_status()
    print("killed the video worker mid-job at", a.get(f"/v1/jobs/{jid}").json()["processed_frames"], "frames")
    d.post(f"/containers/{worker['Id']}/start").raise_for_status()
    j = wait(lambda: (lambda x: x if x["status"] in ("completed", "failed") else None)(a.get(f"/v1/jobs/{jid}").json()),
             300, every=2, what="job finished after reclaim")
    if j["status"] != "completed" or j["attempts"] != 2:
        raise SystemExit(f"FAIL: expected completed on attempt 2, got {j['status']} attempts={j['attempts']} {j['error']}")
    print(f"job-reclaim OK: completed on attempt {j['attempts']}, {j['processed_frames']} frames")


def triton_restart(project: str) -> None:
    """Triton comes back empty (explicit model control): the API's watchdog reloads the
    active versions, and the shards re-register their shared memory (or restart)."""
    a, d = api(), docker()
    ids = []
    for i in range(2):
        r = a.post("/v1/cameras", json={"name": f"drill-t{i}", "url": f"rtsp://mediamtx:8554/sub{i}", "enabled": True})
        r.raise_for_status()
        ids.append(r.json()["id"])

    def cams_ok():
        now = time.time()
        for cid in ids:
            s = a.get(f"/v1/cameras/{cid}").json()["status"]
            if not s or s["state"] != "streaming" or s["fps"] < 3 or now - s["updated_at"] > 5:
                return False
        return True

    try:
        wait(cams_ok, 120, what="cameras streaming before the restart")
        triton = containers(d, project, "triton")[0]
        t0 = time.time()
        d.post(f"/containers/{triton['Id']}/restart", params={"t": 5}, timeout=120).raise_for_status()
        wait(lambda: a.get("/health/ready").status_code == 200, 180, every=2, what="ready again")
        print(f"ready again {time.time() - t0:.0f}s after the Triton restart")
        # The fps window spans 10 s, so give it time to refill with post-restart frames.
        time.sleep(12)
        wait(cams_ok, 180, every=2, what="cameras inferring again")
        print(f"cameras inferring again {time.time() - t0:.0f}s after the restart")
    finally:
        for cid in ids:
            a.delete(f"/v1/cameras/{cid}")
    print("triton-restart OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("drill", choices=["shard-failover", "job-reclaim", "triton-restart"])
    ap.add_argument("--project", default="yp-test")
    args = ap.parse_args()
    {"shard-failover": shard_failover, "job-reclaim": job_reclaim,
     "triton-restart": triton_restart}[args.drill](args.project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
