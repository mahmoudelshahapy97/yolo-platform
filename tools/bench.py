"""Capacity benchmark: N test cameras at T fps, measured through Prometheus.

Runs inside the compose network (API image: httpx is there), e.g.

    docker compose run --rm --no-deps -v ./tools:/tools:ro -v ./artifacts:/artifacts \
        --entrypoint python api /tools/bench.py --counts 10 25 50 --label nvdec

For every camera count it:
  1. replaces all cameras named bench-* with N new ones on the mediamtx test rig
     (rtsp://mediamtx:8554/<stream><i % 6>), target fps --fps, one presence rule;
  2. waits --warmup seconds, then measures over --window seconds:
       inferred fps (total, per camera min / median), dropped (stale) frames,
       decode-to-result latency p50/p95, Triton batch size and queue time,
       GPU SM / NVDEC utilisation and VRAM (DCGM), host CPU busy % and load
       (/proc/stat and /proc/loadavg are host-wide inside a container);
  3. a count "keeps up" when every camera gets >= 90% of its target fps and drops < 5%.
Writes artifacts/bench/<label>-<timestamp>.json and .md, and removes its cameras
afterwards (unless --keep).
"""

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

API = os.environ.get("BENCH_API", "http://api:8000")
PROM = os.environ.get("BENCH_PROMETHEUS", "http://prometheus:9090")
KEY = os.environ.get("ADMIN_API_KEY", "")


def cpu_sample():
    with open("/proc/stat") as f:
        parts = [int(x) for x in f.readline().split()[1:]]
    idle = parts[3] + parts[4]
    return sum(parts), idle


def prom(client, expr):
    r = client.get(f"{PROM}/api/v1/query", params={"query": expr}, timeout=20)
    r.raise_for_status()
    return r.json()["data"]["result"]


def prom_scalar(client, expr):
    res = prom(client, expr)
    vals = [float(x["value"][1]) for x in res if x["value"][1] not in ("NaN", "+Inf", "-Inf")]
    return round(sum(vals), 3) if vals else None


def bench_cameras(api):
    cams = api.get("/v1/cameras").raise_for_status().json()
    return [c for c in cams if c["name"].startswith("bench-")]


def clear(api):
    for c in bench_cameras(api):
        api.delete(f"/v1/cameras/{c['id']}").raise_for_status()


def create(api, n, fps, stream, model, triggered, tiling="off"):
    config = {"rules": [{"name": "person", "type": "presence", "classes": ["person"], "cooldown_s": 30}],
              "tiling": {"mode": tiling}}
    if triggered:
        config["triggered"] = [{"model": "fire_detection", "every_s": 2.0, "min_confidence": 0.5, "cooldown_s": 60}]
    for i in range(n):
        body = {"name": f"bench-{i:03d}", "url": f"rtsp://mediamtx:8554/{stream}{i % 6}", "target_fps": fps,
                "model": model, "enabled": True, "config": config}
        api.post("/v1/cameras", json=body).raise_for_status()


def measure(api, pc, n, fps, window):
    t0 = time.time()
    c0 = cpu_sample()
    time.sleep(window)
    c1 = cpu_sample()
    total, idle = c1[0] - c0[0], c1[1] - c0[1]
    w = f"{int(window)}s"
    per_cam = prom(pc, f"rate(camera_inferred_frames_total{{camera=~'.+'}}[{w}])")
    owned = {c["id"] for c in bench_cameras(api)}
    rates = sorted(float(x["value"][1]) for x in per_cam if x["metric"].get("camera") in owned)
    rates += [0.0] * (n - len(rates))   # cameras that never produced a frame
    inferred = sum(rates)
    dropped = prom_scalar(pc, f"sum(rate(camera_dropped_frames_total[{w}]))") or 0.0
    statuses = [c.get("status") or {} for c in bench_cameras(api)]
    with open("/proc/loadavg") as f:
        load1 = float(f.read().split()[0])
    result = {
        "cameras": n, "target_fps": fps, "window_s": window, "measured_at": datetime.fromtimestamp(t0, timezone.utc).isoformat(),
        "streaming": sum(1 for s in statuses if s.get("state") == "streaming"),
        "inferred_fps_total": round(inferred, 1),
        "target_fps_total": n * fps,
        "per_camera_fps_min": round(rates[0], 2) if rates else 0.0,
        "per_camera_fps_median": round(statistics.median(rates), 2) if rates else 0.0,
        "dropped_per_s": round(dropped, 1),
        "drop_ratio": round(dropped / (dropped + inferred), 4) if inferred + dropped else None,
        "frame_age_p50_s": prom_scalar(pc, f"histogram_quantile(0.5, sum by (le) (rate(frame_age_seconds_bucket[{w}])))"),
        "frame_age_p95_s": prom_scalar(pc, f"histogram_quantile(0.95, sum by (le) (rate(frame_age_seconds_bucket[{w}])))"),
        "triton_avg_batch": prom_scalar(pc, f"sum(rate(nv_inference_count{{model=~'.*detection.*'}}[{w}])) / sum(rate(nv_inference_exec_count{{model=~'.*detection.*'}}[{w}]))"),
        "triton_queue_ms": prom_scalar(pc, f"sum(rate(nv_inference_queue_duration_us[{w}])) / sum(rate(nv_inference_request_success[{w}])) / 1000"),
        "triton_compute_ms": prom_scalar(pc, f"sum(rate(nv_inference_compute_infer_duration_us[{w}])) / sum(rate(nv_inference_request_success[{w}])) / 1000"),
        "gpu_util_pct": prom_scalar(pc, f"avg(avg_over_time(DCGM_FI_DEV_GPU_UTIL[{w}]))"),
        "nvdec_util_pct": prom_scalar(pc, f"avg(avg_over_time(DCGM_FI_DEV_DEC_UTIL[{w}]))"),
        "vram_used_mib": prom_scalar(pc, "max(DCGM_FI_DEV_FB_USED)"),
        "host_cpu_busy_pct": round(100.0 * (1 - idle / total), 1) if total else None,
        "host_load1": load1,
        "reconnects": sum(int(s.get("reconnects") or 0) for s in statuses),
    }
    result["keeps_up"] = bool(result["per_camera_fps_min"] >= 0.9 * fps and (result["drop_ratio"] or 0) < 0.05
                              and result["streaming"] == n)
    return result


def to_markdown(label, meta, rows):
    cols = [("cameras", "Cameras"), ("inferred_fps_total", "Inferred fps"), ("target_fps_total", "Target fps"),
            ("per_camera_fps_min", "Min fps/cam"), ("per_camera_fps_median", "Median fps/cam"), ("drop_ratio", "Drop ratio"),
            ("frame_age_p95_s", "Latency p95 (s)"), ("triton_avg_batch", "Avg batch"), ("gpu_util_pct", "GPU %"),
            ("nvdec_util_pct", "NVDEC %"), ("vram_used_mib", "VRAM MiB"), ("host_cpu_busy_pct", "Host CPU %"),
            ("host_load1", "Load"), ("keeps_up", "Keeps up")]
    out = [f"# Benchmark: {label}", "", "```", json.dumps(meta, indent=1), "```", "",
           "| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(k) is None else str(r.get(k)) for k, _ in cols) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=int, nargs="+", default=[10, 25, 50, 75, 100])
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--stream", choices=["sub", "main"], default="sub")
    ap.add_argument("--model", default="yolo26n_detection")
    ap.add_argument("--triggered", action="store_true", help="also run fire detection every 2 s per camera")
    ap.add_argument("--warmup", type=float, default=45)
    ap.add_argument("--window", type=float, default=60)
    ap.add_argument("--label", default="bench")
    ap.add_argument("--stop-after-fail", type=int, default=1, help="stop after this many counts that do not keep up")
    ap.add_argument("--tiling", choices=["off", "auto", "on"], default="off",
                    help="cameras' tiling mode (auto tiles main streams: 5 inferences per frame)")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default="/artifacts/bench")
    args = ap.parse_args()

    api = httpx.Client(base_url=API, headers={"X-API-Key": KEY}, timeout=30)
    pc = httpx.Client()
    meta = {"label": args.label, "fps": args.fps, "stream": args.stream, "model": args.model, "triggered": args.triggered, "tiling": args.tiling,
            "warmup_s": args.warmup, "window_s": args.window, "started": datetime.now(timezone.utc).isoformat(),
            "shards": prom_scalar(pc, "count(up{job='camera-shard'} == 1)"),
            "backend": os.environ.get("BENCH_BACKEND_NOTE", "")}
    rows, fails = [], 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for n in args.counts:
            clear(api)
            time.sleep(12)           # leases released, streams closed
            create(api, n, args.fps, args.stream, args.model, args.triggered, args.tiling)
            print(f"[{args.label}] {n} cameras: warming up {args.warmup:.0f}s", flush=True)
            time.sleep(args.warmup)
            r = measure(api, pc, n, args.fps, args.window)
            rows.append(r)
            print(json.dumps(r), flush=True)
            (out_dir / f"{args.label}-{stamp}.json").write_text(json.dumps({"meta": meta, "results": rows}, indent=1))
            (out_dir / f"{args.label}-{stamp}.md").write_text(to_markdown(args.label, meta, rows))
            if not r["keeps_up"]:
                fails += 1
                if fails >= args.stop_after_fail:
                    break
    finally:
        if not args.keep:
            clear(api)
    print(f"report: {out_dir}/{args.label}-{stamp}.md")


if __name__ == "__main__":
    main()
