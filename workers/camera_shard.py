"""Camera shard: decodes a fair share of the enabled cameras and runs them through Triton.

    python -m workers.camera_shard

Layout of one shard process (one CUDA context for all of it):

  decode threads (1 per camera)  NVDEC or CPU, keep the latest sampled, letterboxed frame
           |  take(): copy into a shared-memory batch slot (latest frame wins, stale = dropped)
  inference pipelines (2)        batch fresh frames of cameras sharing a model -> Triton
           |                     (Triton's dynamic batcher merges the shards' requests further)
  post-processing (inline)       decode -> per-camera ByteTrack -> rules -> events
           |                     classify crops / queue triggered models
  triggered-model thread         fire / PPE / ... on a schedule or when a rule fires
  behaviour thread               pose cameras with ``shoplifting``: judges people's pose windows
                                 with the sequence model (shared/yolo_platform_shared/behavior.py)
  control loop (main thread)     leases (scheduler.py), config changes, status, heartbeats

The primary model is a detector or a pose model (its rows carry 17 keypoints; the
shared-memory output region is sized for those).

Events go to the Redis Stream ``events`` (the event processor stores them).
Status goes to ``camera:status:<id>`` every 2 s; metrics on :9100/metrics.
"""

import json
import logging
import os
import queue
import signal
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import tritonclient.grpc as grpcclient
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from redis import Redis

from yolo_platform_shared.behavior import SHOPLIFTING_CLASS, Person, ShopliftingParams, ShopliftingTracker, keypoints_xyn
from yolo_platform_shared.camera_config import CameraConfig
from yolo_platform_shared.decode import decode_detections, decode_pose
from yolo_platform_shared.preprocess import LetterboxMeta, classify_crop, letterbox

from .decode.base import FrameSource, Sample
from .rules import Event, RulesEngine, TrackedObject, rule_by_name
from .scheduler import LeaseScheduler
from .tracking import ByteTracker
from .triton_io import ActiveModels, ShmSlot, infer_plain, is_overload, model_input_name, model_outputs

logger = logging.getLogger("shard")

EVENT_STREAM = "events"
EVENT_STREAM_MAXLEN = 500_000
DETECT_SIZE = 640
SLOT_BATCH = 16
PRIMARY_TASKS = ("detect", "pose")

# ---------------- metrics ----------------
M_ACTIVE = Gauge("active_cameras", "Cameras owned by this shard", ["shard"])
M_STREAMING = Gauge("streaming_cameras", "Cameras currently streaming", ["shard"])
M_FPS = Gauge("camera_fps", "Inferred frames per second (last 10 s)", ["camera"])
M_INFERRED = Counter("camera_inferred_frames_total", "Frames inferred", ["camera"])
M_DROPPED = Counter("camera_dropped_frames_total", "Sampled frames dropped before inference (stale)", ["camera"])
M_DECODED = Counter("camera_decoded_frames_total", "Frames decoded", ["camera"])
M_RECONNECTS = Counter("camera_reconnects_total", "Stream reconnects", ["camera"])
M_BATCH = Histogram("shard_batch_size", "Frames per Triton request", ["model"], buckets=(1, 2, 4, 6, 8, 12, 16))
M_INFER = Histogram("shard_infer_seconds", "Triton request latency seen by the shard", ["model"],
                    buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1, 2))
M_AGE = Histogram("frame_age_seconds", "Decode-to-result latency", buckets=(0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5))
M_POST = Histogram("shard_postprocess_seconds", "Post-processing per batch",
                   buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25))
M_EVENTS = Counter("events_total", "Events emitted", ["type"])
M_ERRORS = Counter("shard_infer_errors_total", "Failed Triton requests", ["model", "reason"])
M_TRIGGERED = Counter("triggered_inferences_total", "Triggered-model inferences", ["model"])
M_TRIG_DROP = Counter("triggered_dropped_total", "Triggered-model requests dropped (queue full)", ["model"])
M_BEHAVIOR = Counter("behavior_windows_total", "Pose windows judged by a sequence model", ["model"])
M_BEHAVIOR_DROP = Counter("behavior_dropped_total", "Pose windows dropped before judging (queue full)", ["model"])


@dataclass
class Settings:
    redis_url: str = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    triton_url: str = os.environ.get("TRITON_URL", "triton:8001")
    backend: str = os.environ.get("DECODE_BACKEND", "nvdec")
    max_cameras: int = int(os.environ.get("SHARD_MAX_CAMERAS", "40"))
    pipelines: int = int(os.environ.get("SHARD_PIPELINES", "2"))
    metrics_port: int = int(os.environ.get("METRICS_PORT", "9100"))
    shard_id: str = os.environ.get("SHARD_ID") or f"{socket.gethostname()}-{os.getpid()}"
    lease_ms: int = int(os.environ.get("LEASE_MS", "15000"))
    stall_s: float = float(os.environ.get("STALL_SECONDS", "20"))


def out_cols(model: dict) -> int:
    """Row width of a box model's output: 6, plus K*3 keypoint values for pose."""
    if model.get("task") == "pose":
        k, d = model.get("kpt_shape") or (17, 3)
        return 6 + k * d
    return 6


def shoplifting_params(config: CameraConfig) -> Optional[ShopliftingParams]:
    c = config.shoplifting
    if c is None:
        return None
    return ShopliftingParams(sample_fps=c.sample_fps, threshold=c.threshold, smooth_window=c.smooth_window,
                             min_window_samples=c.min_window_samples, min_person_height_px=c.min_person_height_px,
                             track_cooldown_s=c.track_cooldown_s, camera_cooldown_s=c.camera_cooldown_s)


def make_source(backend: str, camera_id: str, url: str, fps: float) -> FrameSource:
    if backend == "nvdec":
        from .decode.nvdec import NvdecSource
        return NvdecSource(camera_id, url, fps, DETECT_SIZE)
    from .decode.cpu import CpuSource
    return CpuSource(camera_id, url, fps, DETECT_SIZE)


@dataclass
class CameraRuntime:
    id: str
    spec: dict
    config: CameraConfig
    source: FrameSource
    tracker: ByteTracker
    rules: RulesEngine
    lock: threading.Lock = field(default_factory=threading.Lock)  # tracker + rules are per camera, one batch at a time
    inferred: int = 0
    fps_window: List[Tuple[float, int]] = field(default_factory=list)
    next_periodic: Dict[str, float] = field(default_factory=dict)
    trig_cooldown: Dict[Tuple[str, str], float] = field(default_factory=dict)
    last_events: int = 0
    seen: Dict[str, int] = field(default_factory=lambda: {"decoded": 0, "dropped": 0, "reconnects": 0})
    shoplifting: Optional[ShopliftingTracker] = None

    @property
    def model(self) -> str:
        return self.spec["model"]


@dataclass
class TriggeredJob:
    camera: CameraRuntime
    model: str
    frame: np.ndarray            # letterboxed 640 RGB
    meta: LetterboxMeta
    ts: float
    reason: str
    classes: Optional[List[str]]
    min_confidence: float
    cooldown_s: float


@dataclass
class BehaviorJob:
    camera: CameraRuntime
    model: str
    windows: List[Tuple[int, np.ndarray]]   # (track_id, [120, 17, 12] features)
    ts: float


class Shard:
    def __init__(self, s: Settings) -> None:
        self.s = s
        self.redis = Redis.from_url(s.redis_url, decode_responses=True)
        self.models = ActiveModels(self.redis)
        self.scheduler = LeaseScheduler(self.redis, s.shard_id, s.max_cameras, s.lease_ms)
        self.cameras: Dict[str, CameraRuntime] = {}
        self.cam_lock = threading.Lock()
        self.stop = threading.Event()
        self.triggered: "queue.Queue[TriggeredJob]" = queue.Queue(maxsize=256)
        self.behavior: "queue.Queue[BehaviorJob]" = queue.Queue(maxsize=256)
        self._rr = 0

    # ---------------- camera lifecycle ----------------

    def _start_camera(self, cam_id: str, spec: dict) -> None:
        config = CameraConfig.model_validate(spec.get("config") or {})
        fps = float(spec.get("target_fps", 5))
        src = make_source(self.s.backend, cam_id, spec["url"], fps)
        rt = CameraRuntime(cam_id, spec, config, src, ByteTracker(frame_rate=fps), RulesEngine(config))
        params = shoplifting_params(config)
        rt.shoplifting = ShopliftingTracker(params) if params else None
        with self.cam_lock:
            self.cameras[cam_id] = rt
        src.start()
        logger.info("camera %s: started (%s, %s fps, model %s)", cam_id, self.s.backend, fps, spec["model"])

    def _stop_camera(self, cam_id: str, reason: str) -> None:
        with self.cam_lock:
            rt = self.cameras.pop(cam_id, None)
        if rt is None:
            return
        rt.source.stop()
        self.redis.delete(f"camera:status:{cam_id}")
        # Drop the camera's series so label cardinality does not grow with camera churn.
        for metric in (M_FPS, M_INFERRED, M_DROPPED, M_DECODED, M_RECONNECTS):
            try:
                metric.remove(cam_id)
            except KeyError:
                pass
        logger.info("camera %s: stopped (%s)", cam_id, reason)

    def _apply_changes(self, desired: Dict[str, dict]) -> None:
        """Restart a camera whose stream changed; hot-swap rules otherwise."""
        for cam_id, rt in list(self.cameras.items()):
            spec = desired.get(cam_id)
            if spec is None or spec.get("updated_at") == rt.spec.get("updated_at"):
                continue
            if spec["url"] != rt.spec["url"] or spec["model"] != rt.spec["model"]:
                self._stop_camera(cam_id, "stream or model changed")
                self._start_camera(cam_id, spec)
                continue
            config = CameraConfig.model_validate(spec.get("config") or {})
            with rt.lock:
                rt.spec, rt.config = spec, config
                rt.rules = RulesEngine(config)
                params = shoplifting_params(config)
                if params is None:
                    rt.shoplifting = None
                elif rt.shoplifting is None:
                    rt.shoplifting = ShopliftingTracker(params)
                else:
                    rt.shoplifting.set_params(params)   # keep people's windows
                rt.source.set_fps(float(spec["target_fps"]))
                rt.tracker.set_frame_rate(float(spec["target_fps"]))
            logger.info("camera %s: config updated", cam_id)

    def _supervise(self) -> None:
        """Replace sources that stopped producing frames (a hung RTSP read cannot be interrupted)."""
        now = time.time()
        for cam_id, rt in list(self.cameras.items()):
            st = rt.source.stats
            if st.state == "streaming" and st.last_frame_at and now - st.last_frame_at > self.s.stall_s:
                logger.warning("camera %s: no frame for %.0fs; replacing its reader", cam_id, now - st.last_frame_at)
                spec = rt.spec
                self._stop_camera(cam_id, "stalled")
                self._start_camera(cam_id, spec)

    # ---------------- inference pipelines ----------------

    def _fresh_by_model(self) -> Dict[str, List[CameraRuntime]]:
        groups: Dict[str, List[CameraRuntime]] = {}
        with self.cam_lock:
            cams = list(self.cameras.values())
        if not cams:
            return groups
        self._rr = (self._rr + 1) % len(cams)
        for rt in cams[self._rr:] + cams[:self._rr]:   # rotate, so no camera is always first
            if rt.source.has_fresh():
                groups.setdefault(rt.model, []).append(rt)
        return groups

    def _pipeline(self, index: int) -> None:
        client = grpcclient.InferenceServerClient(self.s.triton_url)
        slot = self._new_slot(client, index)
        failures = 0
        try:
            while not self.stop.is_set():
                groups = self._fresh_by_model()
                if not groups:
                    time.sleep(0.003)
                    continue
                model_name, cams = max(groups.items(), key=lambda kv: len(kv[1]))
                model = self.models.get(model_name)
                if model is None or model["imgsz"] != DETECT_SIZE or model.get("task", "detect") not in PRIMARY_TASKS:
                    time.sleep(0.05)
                    continue
                batch_max = min(SLOT_BATCH, model["max_batch"])
                taken: List[Tuple[CameraRuntime, Sample]] = []
                for rt in cams:
                    if len(taken) >= batch_max:
                        break
                    sample = rt.source.take(slot.inputs[len(taken)])
                    if sample is not None:
                        taken.append((rt, sample))
                if not taken:
                    continue
                t0 = time.perf_counter()
                try:
                    out = slot.infer(model["serve_name"], 1, len(taken), out_cols=out_cols(model))
                    failures = 0
                except Exception as exc:
                    overload = is_overload(exc)
                    M_ERRORS.labels(model_name, "overload" if overload else "error").inc()
                    logger.warning("pipeline %d: %s inference failed for %d frames: %s", index, model_name,
                                   len(taken), exc)
                    failures += 0 if overload else 1
                    if failures in (10, 30):
                        # Triton restarted and forgot our shared-memory registration: register again.
                        slot.close()
                        slot = self._new_slot(client, index)
                    elif failures >= 60:
                        # Still failing (e.g. Triton came back in a new IPC namespace): let Docker
                        # restart this shard; its leases expire and are re-claimed.
                        logger.error("pipeline %d: %d consecutive failures; exiting so the shard restarts",
                                     index, failures)
                        os._exit(1)
                    time.sleep(0.1 if overload else 0.5)
                    continue
                M_INFER.labels(model_name).observe(time.perf_counter() - t0)
                M_BATCH.labels(model_name).observe(len(taken))
                t1 = time.perf_counter()
                for i, (rt, sample) in enumerate(taken):
                    try:
                        self._post(rt, sample, out[i], slot.inputs[i], model, client)
                    except Exception:
                        logger.exception("camera %s: post-processing failed", rt.id)
                M_POST.observe(time.perf_counter() - t1)
        finally:
            slot.close()
            client.close()

    def _new_slot(self, client, index: int) -> ShmSlot:
        while not self.stop.is_set():
            try:
                slot = ShmSlot(client, SLOT_BATCH, DETECT_SIZE)
                logger.info("pipeline %d ready (shared-memory slot %s)", index, slot.in_name)
                return slot
            except Exception as exc:
                logger.warning("pipeline %d: cannot register shared memory with Triton yet: %s", index, exc)
                time.sleep(2)
        raise SystemExit(0)

    def _post(self, rt: CameraRuntime, sample: Sample, out: np.ndarray, frame: np.ndarray, model: dict,
              client) -> None:
        now = time.time()
        M_AGE.observe(now - sample.ts)
        with rt.lock:
            config = rt.config
            names = model["names"]
            keep = None
            if config.classes:
                wanted = set(config.classes)
                keep = [i for i, n in names.items() if n in wanted]
            low = min(config.detect_confidence, 0.1)
            if model.get("task") == "pose":
                dets = decode_pose(out, sample.meta, names, tuple(model.get("kpt_shape") or (17, 3)), conf=low,
                                   classes=keep)
            else:
                dets = decode_detections(out, sample.meta, names, conf=low, classes=keep)
            # ByteTrack wants the low-confidence detections too (second association stage).
            boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32).reshape(-1, 4)
            scores = np.array([d.confidence for d in dets], dtype=np.float32)
            cls = np.array([d.class_id for d in dets], dtype=np.int64)
            matches = rt.tracker.update(boxes, scores, cls)
            objects = [TrackedObject(tid, dets[di].class_name, dets[di].confidence,
                                     (dets[di].x1, dets[di].y1, dets[di].x2, dets[di].y2), dets[di].keypoints)
                       for di, tid in matches if dets[di].confidence >= config.detect_confidence]
            events = rt.rules.evaluate(objects, sample.meta.orig_w, sample.meta.orig_h, sample.ts)
            due = []
            if rt.shoplifting is not None and config.shoplifting is not None:
                w, h = sample.meta.orig_w, sample.meta.orig_h
                people = [Person(o.track_id, o.box, keypoints_xyn(o.keypoints, w, h), o.confidence)
                          for o in objects if o.keypoints is not None
                          and o.confidence >= config.shoplifting.min_confidence]
                due = rt.shoplifting.update(people, sample.ts)
            rt.inferred += 1
            rt.fps_window.append((now, rt.inferred))
            while rt.fps_window and now - rt.fps_window[0][0] > 10:
                rt.fps_window.pop(0)
        M_INFERRED.labels(rt.id).inc()
        if due:
            try:
                self.behavior.put_nowait(BehaviorJob(rt, config.shoplifting.model, due, sample.ts))
            except queue.Full:
                M_BEHAVIOR_DROP.labels(config.shoplifting.model).inc(len(due))

        for ev in events:
            rule = rule_by_name(config, ev.rule)
            if rule is not None and rule.classify and ev.box is not None:
                ev.data["classification"] = self._classify(client, rule.classify.model, rule.classify.top_k,
                                                           frame, sample.meta, ev.box)
            self._emit(rt, ev)
            for t in config.triggered:
                if t.on_rule == ev.rule:
                    self._queue_triggered(rt, t, frame, sample, f"rule:{ev.rule}")
        for t in config.triggered:
            if t.every_s and now >= rt.next_periodic.get(t.model, 0.0):
                rt.next_periodic[t.model] = now + t.every_s
                self._queue_triggered(rt, t, frame, sample, "periodic")

    def _classify(self, client, model_name: str, top_k: int, frame_lb: np.ndarray, meta: LetterboxMeta, box) -> list:
        model = self.models.get(model_name)
        if model is None:
            return []
        # Crop from the letterboxed frame: map the original-pixel box into model space.
        x1, y1, x2, y2 = (box[0] * meta.ratio + meta.pad_left, box[1] * meta.ratio + meta.pad_top,
                          box[2] * meta.ratio + meta.pad_left, box[3] * meta.ratio + meta.pad_top)
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame_lb.shape[1], int(round(x2))), min(frame_lb.shape[0], int(round(y2)))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return []
        crop = classify_crop(np.ascontiguousarray(frame_lb[y1:y2, x1:x2, ::-1]), model["imgsz"])
        try:
            probs = infer_plain(client, model["serve_name"], 1, crop[None])[0]
        except Exception as exc:
            M_ERRORS.labels(model_name, "error").inc()
            logger.warning("classify %s failed: %s", model_name, exc)
            return []
        idx = np.argsort(-probs)[:top_k]
        return [{"class_id": int(i), "class_name": model["names"].get(int(i), str(i)), "confidence": round(float(probs[i]), 4)}
                for i in idx]

    def _queue_triggered(self, rt: CameraRuntime, t, frame: np.ndarray, sample: Sample, reason: str) -> None:
        job = TriggeredJob(rt, t.model, frame.copy(), sample.meta, sample.ts, reason, t.classes, t.min_confidence,
                           t.cooldown_s)
        try:
            self.triggered.put_nowait(job)
        except queue.Full:
            M_TRIG_DROP.labels(t.model).inc()

    def _triggered_loop(self) -> None:
        client = grpcclient.InferenceServerClient(self.s.triton_url)
        while not self.stop.is_set():
            try:
                first = self.triggered.get(timeout=0.5)
            except queue.Empty:
                continue
            jobs = [first]
            while len(jobs) < 16:
                try:
                    jobs.append(self.triggered.get_nowait())
                except queue.Empty:
                    break
            by_model: Dict[str, List[TriggeredJob]] = {}
            for j in jobs:
                by_model.setdefault(j.model, []).append(j)
            for model_name, items in by_model.items():
                model = self.models.get(model_name)
                if model is None or model["task"] not in PRIMARY_TASKS:
                    continue
                if model["imgsz"] == DETECT_SIZE:
                    batch, metas = np.stack([j.frame for j in items]), [j.meta for j in items]
                else:
                    # Different input size: re-letterbox the visible area of the 640 frame.
                    batch, metas = self._reletterbox(items, model["imgsz"])
                try:
                    out = np.concatenate([infer_plain(client, model["serve_name"], 1, batch[k:k + model["max_batch"]])
                                          for k in range(0, len(batch), model["max_batch"])])
                except Exception as exc:
                    M_ERRORS.labels(model_name, "overload" if is_overload(exc) else "error").inc()
                    logger.warning("triggered %s failed: %s", model_name, exc)
                    continue
                M_TRIGGERED.labels(model_name).inc(len(items))
                for j, o, m in zip(items, out, metas):
                    self._triggered_events(j, model, o, m)
        client.close()

    def _behavior_loop(self) -> None:
        """Judge pose windows with the camera's sequence model, batched across cameras."""
        client = grpcclient.InferenceServerClient(self.s.triton_url)
        while not self.stop.is_set():
            try:
                first = self.behavior.get(timeout=0.5)
            except queue.Empty:
                continue
            jobs = [first]
            while len(jobs) < 32:
                try:
                    jobs.append(self.behavior.get_nowait())
                except queue.Empty:
                    break
            by_model: Dict[str, List[BehaviorJob]] = {}
            for j in jobs:
                by_model.setdefault(j.model, []).append(j)
            for model_name, items in by_model.items():
                model = self.models.get(model_name)
                if model is None or model.get("task") != "sequence":
                    M_BEHAVIOR_DROP.labels(model_name).inc(sum(len(j.windows) for j in items))
                    continue
                owners = [(j, tid) for j in items for tid, _ in j.windows]
                batch = np.stack([w for j in items for _, w in j.windows]).astype(np.float32)
                try:
                    probs = np.concatenate([
                        infer_plain(client, model["serve_name"], 1, batch[k:k + model["max_batch"]],
                                    input_name=model_input_name(model), output=model_outputs(model)[0])
                        for k in range(0, len(batch), model["max_batch"])])
                except Exception as exc:
                    M_ERRORS.labels(model_name, "overload" if is_overload(exc) else "error").inc()
                    logger.warning("behaviour %s failed for %d windows: %s", model_name, len(batch), exc)
                    continue
                M_BEHAVIOR.labels(model_name).inc(len(batch))
                for (j, tid), p in zip(owners, probs):
                    rt = j.camera
                    with rt.lock:
                        if rt.shoplifting is None:
                            continue
                        alert = rt.shoplifting.judge(tid, float(p[SHOPLIFTING_CLASS]), j.ts)
                    if alert is None:
                        continue
                    self._emit(rt, Event(
                        rule="shoplifting", type="shoplifting", ts=j.ts, track_id=alert.track_id, class_name="person",
                        confidence=alert.probability, box=alert.box,
                        data={"probability": round(alert.probability, 4), "peak": round(alert.peak, 4),
                              "model": model_name, "model_version": model["version"],
                              "window_samples": alert.real_samples, "alert": alert.alerts,
                              "tracked_s": round(alert.ts - alert.first_seen, 1)}))
        client.close()

    @staticmethod
    def _reletterbox(items: List[TriggeredJob], size: int):
        batch, metas = [], []
        for j in items:
            m = j.meta
            w, h = int(round(m.orig_w * m.ratio)), int(round(m.orig_h * m.ratio))
            visible = np.ascontiguousarray(j.frame[m.pad_top:m.pad_top + h, m.pad_left:m.pad_left + w, ::-1])
            lb, m2 = letterbox(visible, size)
            # Compose: model2 space -> visible (640-scale) -> original pixels.
            metas.append(LetterboxMeta(orig_w=m.orig_w, orig_h=m.orig_h, ratio=m2.ratio * m.ratio,
                                       pad_left=m2.pad_left, pad_top=m2.pad_top))
            batch.append(lb)
        return np.stack(batch), metas

    def _triggered_events(self, j: TriggeredJob, model: dict, out: np.ndarray, meta: LetterboxMeta) -> None:
        dets = decode_detections(out, meta, model["names"], conf=j.min_confidence)
        if j.classes:
            dets = [d for d in dets if d.class_name in j.classes]
        rt = j.camera
        for d in dets:
            key = (j.model, d.class_name)
            last = rt.trig_cooldown.get(key)
            if last is not None and j.ts - last < j.cooldown_s:
                continue
            rt.trig_cooldown[key] = j.ts
            self._emit(rt, Event(rule=j.model, type="model_detection", ts=j.ts, track_id=None, class_name=d.class_name,
                                 confidence=d.confidence, box=(d.x1, d.y1, d.x2, d.y2),
                                 data={"model": j.model, "model_version": model["version"], "trigger": j.reason}))

    def _emit(self, rt: CameraRuntime, ev: Event) -> None:
        payload = {"camera_id": rt.id, "tenant_id": rt.spec.get("tenant_id"), "ts": ev.ts, "type": ev.type,
                   "rule": ev.rule, "track_id": ev.track_id, "class_name": ev.class_name,
                   "confidence": round(ev.confidence, 4) if ev.confidence is not None else None,
                   "bbox": {k: round(v, 1) for k, v in zip(("x1", "y1", "x2", "y2"), ev.box)} if ev.box else None,
                   "data": ev.data, "shard": self.s.shard_id}
        self.redis.xadd(EVENT_STREAM, {"event": json.dumps(payload)}, maxlen=EVENT_STREAM_MAXLEN, approximate=True)
        M_EVENTS.labels(ev.type).inc()

    # ---------------- status ----------------

    def _report(self) -> None:
        now = time.time()
        pipe = self.redis.pipeline()
        streaming = 0
        for cam_id, rt in list(self.cameras.items()):
            st = rt.source.stats
            win = rt.fps_window
            fps = (win[-1][1] - win[0][1]) / (win[-1][0] - win[0][0]) if len(win) > 1 and win[-1][0] > win[0][0] else 0.0
            streaming += st.state == "streaming"
            M_FPS.labels(cam_id).set(round(fps, 2))
            for name, metric, value in (("decoded", M_DECODED, st.decoded), ("dropped", M_DROPPED, st.dropped),
                                        ("reconnects", M_RECONNECTS, st.reconnects)):
                delta = value - rt.seen[name]
                if delta > 0:
                    metric.labels(cam_id).inc(delta)
                rt.seen[name] = value
            status = {"state": st.state, "backend": rt.source.backend, "shard": self.s.shard_id,
                      "fps": round(fps, 2), "target_fps": rt.source.target_fps, "inferred": rt.inferred,
                      "decoded": st.decoded, "dropped": st.dropped, "reconnects": st.reconnects,
                      "width": st.width, "height": st.height, "codec": st.codec, "error": st.error,
                      "last_frame_age_s": round(now - st.last_frame_at, 2) if st.last_frame_at else None,
                      "updated_at": now}
            key = f"camera:status:{cam_id}"
            pipe.hset(key, mapping={k: json.dumps(v) for k, v in status.items()})
            pipe.expire(key, 10)
        pipe.execute()
        M_ACTIVE.labels(self.s.shard_id).set(len(self.cameras))
        M_STREAMING.labels(self.s.shard_id).set(streaming)

    # ---------------- main ----------------

    def run(self) -> None:
        start_http_server(self.s.metrics_port)
        if self.s.backend == "nvdec":
            from .decode.nvdec import shard_context
            shard_context()
        threads = [threading.Thread(target=self._pipeline, args=(i,), name=f"pipe-{i}", daemon=True)
                   for i in range(self.s.pipelines)]
        threads.append(threading.Thread(target=self._triggered_loop, name="triggered", daemon=True))
        threads.append(threading.Thread(target=self._behavior_loop, name="behavior", daemon=True))
        for t in threads:
            t.start()
        logger.info("shard %s up: backend=%s max_cameras=%d pipelines=%d", self.s.shard_id, self.s.backend,
                    self.s.max_cameras, self.s.pipelines)
        last_report = 0.0
        while not self.stop.is_set():
            try:
                a = self.scheduler.tick()
                for cam in a.stop:
                    self._stop_camera(cam, "lease released")
                for cam in a.start:
                    self._start_camera(cam, a.desired[cam])
                self._apply_changes(a.desired)
                self._supervise()
                if time.time() - last_report >= 2.0:
                    self._report()
                    last_report = time.time()
            except Exception:
                logger.exception("control loop error")
            self.stop.wait(2.0)
        self.shutdown()

    def shutdown(self) -> None:
        logger.info("shard %s: shutting down", self.s.shard_id)
        with self.cam_lock:
            ids = list(self.cameras)
        for cam in ids:
            self._stop_camera(cam, "shutdown")
        self.scheduler.release_all()


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    shard = Shard(Settings())
    signal.signal(signal.SIGTERM, lambda *_: shard.stop.set())
    signal.signal(signal.SIGINT, lambda *_: shard.stop.set())
    shard.run()


if __name__ == "__main__":
    main()
