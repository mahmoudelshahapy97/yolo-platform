"""Generate monitoring/grafana/dashboards/platform.json (stdlib only).

    python3 monitoring/gen_dashboard.py
"""

import json
from pathlib import Path

OUT = Path(__file__).parent / "grafana" / "dashboards" / "platform.json"
DS = {"type": "prometheus", "uid": "prometheus"}

panels = []
_y = 0


def row(title):
    global _y
    panels.append({"type": "row", "title": title, "collapsed": False, "gridPos": {"x": 0, "y": _y, "w": 24, "h": 1},
                   "id": len(panels) + 1, "panels": []})
    _y += 1


def ts(title, targets, x, w=8, h=8, unit="short", stack=False):
    panels.append({
        "type": "timeseries", "title": title, "datasource": DS, "id": len(panels) + 1,
        "gridPos": {"x": x, "y": _y, "w": w, "h": h},
        "fieldConfig": {"defaults": {"unit": unit, "custom": {"lineWidth": 2, "fillOpacity": 10,
                                                              "stacking": {"mode": "normal" if stack else "none"}}},
                        "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
        "targets": [{"datasource": DS, "expr": e, "legendFormat": l, "refId": chr(65 + i)} for i, (e, l) in enumerate(targets)],
    })


def stat(title, expr, x, w=4, h=4, unit="short"):
    panels.append({
        "type": "stat", "title": title, "datasource": DS, "id": len(panels) + 1,
        "gridPos": {"x": x, "y": _y, "w": w, "h": h},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value", "graphMode": "area"},
        "targets": [{"datasource": DS, "expr": expr, "refId": "A"}],
    })


def nl(h):
    global _y
    _y += h


row("Overview")
stat("Cameras owned", "sum(active_cameras)", 0)
stat("Cameras streaming", "sum(streaming_cameras)", 4)
stat("Frames inferred / s", "sum(rate(camera_inferred_frames_total[1m]))", 8)
stat("Dropped (stale) / s", "sum(rate(camera_dropped_frames_total[1m]))", 12)
stat("GPU util", "avg(DCGM_FI_DEV_GPU_UTIL)", 16, unit="percent")
stat("NVDEC util", "avg(DCGM_FI_DEV_DEC_UTIL)", 20, unit="percent")
nl(4)

row("GPU")
ts("GPU / NVDEC utilisation", [("avg(DCGM_FI_DEV_GPU_UTIL)", "SM"), ("avg(DCGM_FI_DEV_DEC_UTIL)", "NVDEC"),
                               ("avg(DCGM_FI_DEV_MEM_COPY_UTIL)", "memory copy")], 0, unit="percent")
ts("GPU memory", [("sum(DCGM_FI_DEV_FB_USED) * 1024 * 1024", "used"),
                  ("sum(DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE) * 1024 * 1024", "total")], 8, unit="bytes")
ts("Temperature / power", [("avg(DCGM_FI_DEV_GPU_TEMP)", "temp C"), ("avg(DCGM_FI_DEV_POWER_USAGE)", "power W")], 16)
nl(8)

row("Triton")
ts("Inferences / s by model", [("sum by (model) (rate(nv_inference_count[1m]))", "{{model}}")], 0, stack=True)
ts("Avg batch size (inferences / executions)",
   [("sum by (model) (rate(nv_inference_count[1m])) / sum by (model) (rate(nv_inference_exec_count[1m]))", "{{model}}")], 8)
ts("Queue and compute time per request",
   [("sum by (model) (rate(nv_inference_queue_duration_us[1m])) / sum by (model) (rate(nv_inference_request_success[1m])) / 1e6", "queue {{model}}"),
    ("sum by (model) (rate(nv_inference_compute_infer_duration_us[1m])) / sum by (model) (rate(nv_inference_request_success[1m])) / 1e6", "compute {{model}}")],
   16, unit="s")
nl(8)
ts("Pending requests", [("sum by (model) (nv_inference_pending_request_count)", "{{model}}")], 0)
ts("Failures / s", [("sum by (model) (rate(nv_inference_request_failure[1m]))", "{{model}}")], 8)
ts("Model load time", [("max by (model) (nv_model_load_duration_secs)", "{{model}}")], 16, unit="s")
nl(8)

row("API")
ts("Image inference latency", [
    (f"histogram_quantile({q}, sum by (le) (rate(inference_latency_seconds_bucket[5m])))", f"p{int(q * 100)}")
    for q in (0.5, 0.95, 0.99)], 0, unit="s")
ts("Requests / s", [("sum by (model, status) (rate(inference_requests_total[1m]))", "{{model}} {{status}}"),
                    ("sum by (reason) (rate(inference_rejected_total[1m]))", "rejected {{reason}}")], 8)
ts("Stage time p95", [("histogram_quantile(0.95, sum by (le, stage) (rate(inference_stage_seconds_bucket[5m])))", "{{stage}}")],
   16, unit="s")
nl(8)

row("Cameras")
ts("Inferred fps per camera", [("camera_fps", "{{camera}}")], 0, w=12)
ts("Dropped frames / s per camera", [("rate(camera_dropped_frames_total[1m])", "{{camera}}")], 12, w=12)
nl(8)
ts("Decode-to-result latency", [
    (f"histogram_quantile({q}, sum by (le) (rate(frame_age_seconds_bucket[5m])))", f"p{int(q * 100)}") for q in (0.5, 0.95, 0.99)],
   0, unit="s")
ts("Shard batch size (avg)", [("sum by (instance) (rate(shard_batch_size_sum[1m])) / sum by (instance) (rate(shard_batch_size_count[1m]))", "{{instance}}")], 8)
ts("Reconnects / min", [("sum by (camera) (increase(camera_reconnects_total[1m])) > 0", "{{camera}}")], 16)
nl(8)

row("Events and jobs")
ts("Events / s by type", [("sum by (type) (rate(events_total[1m]))", "{{type}}"),
                          ("rate(events_stored_total[1m])", "stored")], 0)
ts("Triggered-model inferences / s", [("sum by (model) (rate(triggered_inferences_total[1m]))", "{{model}}"),
                                      ("sum by (model) (rate(triggered_dropped_total[1m]))", "dropped {{model}}")], 8)
ts("Video jobs", [("sum(video_jobs_active)", "active"), ("sum by (status) (increase(video_jobs_total[5m]))", "{{status}} /5m"),
                  ("sum(rate(video_job_frames_total[1m]))", "frames/s")], 16)
nl(8)

dashboard = {
    "uid": "yolo-platform", "title": "YOLO Inference Platform", "schemaVersion": 39, "version": 1,
    "time": {"from": "now-1h", "to": "now"}, "refresh": "10s", "tags": ["yolo", "triton"],
    "panels": panels, "templating": {"list": []}, "annotations": {"list": []},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(dashboard, indent=1))
print(f"wrote {OUT} ({len(panels)} panels)")
