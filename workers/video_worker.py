"""Video job worker: Redis Stream ``jobs:video`` -> decode, sample, infer, track -> results in object storage.

    python -m workers.video_worker

Durability:
  - consumer group ``video-workers``; an entry is acknowledged only when its job
    reaches a final state (completed / failed / cancelled);
  - entries a dead worker left pending for > CLAIM_IDLE_S are taken over with
    XAUTOCLAIM and retried;
  - each start increments job.attempts; after MAX_ATTEMPTS the job is failed and
    the entry is copied to the dead-letter stream ``jobs:video:dead``.

Output: ``jobs/<id>/detections.jsonl`` (one line per sampled frame; pose models add
``keypoints``, segmentation models ``polygon`` to each detection), a summary in
job.summary, and with annotate=true ``jobs/<id>/annotated.mp4`` (H.264).
"""

import json
import logging
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
from collections import Counter as CountDict
from pathlib import Path

import boto3
import cv2
import numpy as np
import psycopg
import tritonclient.grpc as grpcclient
from botocore.config import Config
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from redis import Redis
from redis.exceptions import ResponseError

from yolo_platform_shared.decode import decode_detections, decode_pose, decode_segment
from yolo_platform_shared.preprocess import letterbox

from .tracking import ByteTracker
from .triton_io import ActiveModels, infer_outputs, model_outputs

logger = logging.getLogger("video-worker")

STREAM, GROUP, DEAD = "jobs:video", "video-workers", "jobs:video:dead"
MAX_ATTEMPTS = int(os.environ.get("JOB_MAX_ATTEMPTS", "3"))
CLAIM_IDLE_S = int(os.environ.get("JOB_CLAIM_IDLE_SECONDS", "120"))
HEARTBEAT_S = 20  # in-progress jobs touch their stream entry (XCLAIM to self) so they are not reclaimed

M_JOBS = Counter("video_jobs_total", "Video jobs finished", ["status"])
M_FRAMES = Counter("video_job_frames_total", "Frames inferred for video jobs")
M_ACTIVE = Gauge("video_jobs_active", "Jobs being processed by this worker")
M_SECONDS = Histogram("video_job_seconds", "Job wall time", buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600))

# COCO-17 skeleton (keypoint index pairs), as Ultralytics draws it.
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
            (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
KPT_DRAW_CONF = 0.5

PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72),
           (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0)]


class JobCancelled(Exception):
    pass


class Worker:
    def __init__(self) -> None:
        self.redis = Redis.from_url(os.environ.get("REDIS_URL", "redis://redis:6379/0"), decode_responses=True)
        self.db_url = os.environ["DATABASE_URL"]
        self.triton = grpcclient.InferenceServerClient(os.environ.get("TRITON_URL", "triton:8001"))
        self.models = ActiveModels(self.redis)
        self.bucket = os.environ.get("S3_BUCKET", "yolo-platform")
        self.s3 = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT") or None,
                               aws_access_key_id=os.environ.get("S3_ACCESS_KEY") or None,
                               aws_secret_access_key=os.environ.get("S3_SECRET_KEY") or None,
                               region_name=os.environ.get("S3_REGION", "us-east-1"),
                               config=Config(signature_version="s3v4", s3={"addressing_style": "path"}))
        self.consumer = f"{socket.gethostname()}-{os.getpid()}"
        self.stop = threading.Event()
        try:
            self.redis.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    # ---------- db ----------

    def _db(self):
        return psycopg.connect(self.db_url, autocommit=True)

    def _job(self, conn, job_id: str):
        cur = conn.execute("SELECT id, status, model, params, input_key, attempts FROM job WHERE id = %s", (job_id,))
        return cur.fetchone()

    def _publish(self, job_id: str) -> None:
        self.redis.publish(f"job:{job_id}", "1")

    # ---------- loop ----------

    def run(self) -> None:
        logger.info("consuming %s as %s", STREAM, self.consumer)
        next_claim = 0.0
        while not self.stop.is_set():
            try:
                entries = []
                if time.monotonic() >= next_claim:
                    _, claimed, _ = self.redis.xautoclaim(STREAM, GROUP, self.consumer,
                                                          min_idle_time=CLAIM_IDLE_S * 1000, count=1)
                    entries = claimed
                    next_claim = time.monotonic() + 15
                    if claimed:
                        logger.info("reclaimed %s from a dead consumer", claimed[0][0])
                if not entries:
                    resp = self.redis.xreadgroup(GROUP, self.consumer, {STREAM: ">"}, count=1, block=2000)
                    entries = resp[0][1] if resp else []
                for entry_id, fields in entries:
                    self._handle(entry_id, fields)
            except Exception:
                logger.exception("worker loop error")
                time.sleep(2)

    def _handle(self, entry_id: str, fields: dict) -> None:
        job_id = fields.get("job_id")
        with self._db() as conn:
            row = self._job(conn, job_id) if job_id else None
            if row is None or row[1] in ("completed", "failed", "cancelled"):
                self.redis.xack(STREAM, GROUP, entry_id)
                return
            attempts = row[5] + 1
            if attempts > MAX_ATTEMPTS:
                conn.execute("UPDATE job SET status = 'failed', error = %s, finished_at = now() WHERE id = %s",
                             (f"gave up after {MAX_ATTEMPTS} attempts", job_id))
                self.redis.xadd(DEAD, {"job_id": job_id, "entry_id": entry_id, "reason": "max attempts"})
                self.redis.xack(STREAM, GROUP, entry_id)
                M_JOBS.labels("failed").inc()
                self._publish(job_id)
                return
            conn.execute("""UPDATE job SET status = 'processing', attempts = %s, started_at = now(), error = NULL,
                            processed_frames = 0 WHERE id = %s""", (attempts, job_id))
        self._publish(job_id)
        started = time.monotonic()
        M_ACTIVE.inc()
        beat_stop = threading.Event()
        beat = threading.Thread(target=self._heartbeat, args=(entry_id, beat_stop), daemon=True)
        beat.start()
        try:
            summary = self._process(job_id, row[2], row[3] if isinstance(row[3], dict) else json.loads(row[3]), row[4])
            with self._db() as conn:
                conn.execute("""UPDATE job SET status = 'completed', summary = %s, result_key = %s, output_key = %s,
                                finished_at = now() WHERE id = %s AND status = 'processing'""",
                             (json.dumps(summary["summary"]), summary["result_key"], summary.get("output_key"), job_id))
            M_JOBS.labels("completed").inc()
            self.redis.xack(STREAM, GROUP, entry_id)
        except JobCancelled:
            M_JOBS.labels("cancelled").inc()
            self.redis.xack(STREAM, GROUP, entry_id)
        except Exception as exc:
            logger.exception("job %s attempt %d failed", job_id, attempts)
            with self._db() as conn:
                if attempts >= MAX_ATTEMPTS:
                    conn.execute("UPDATE job SET status = 'failed', error = %s, finished_at = now() WHERE id = %s",
                                 (f"{type(exc).__name__}: {exc}"[:1000], job_id))
                    self.redis.xadd(DEAD, {"job_id": job_id, "entry_id": entry_id, "reason": str(exc)[:500]})
                    self.redis.xack(STREAM, GROUP, entry_id)
                    M_JOBS.labels("failed").inc()
                else:
                    # Back to queued; the entry stays pending and is reclaimed after CLAIM_IDLE_S.
                    conn.execute("UPDATE job SET status = 'queued', error = %s WHERE id = %s",
                                 (f"attempt {attempts}: {type(exc).__name__}: {exc}"[:1000], job_id))
        finally:
            beat_stop.set()
            M_ACTIVE.dec()
            M_SECONDS.observe(time.monotonic() - started)
            self._publish(job_id)

    def _heartbeat(self, entry_id: str, stop: threading.Event) -> None:
        """Reset the pending entry's idle time so a long job is not reclaimed while alive."""
        while not stop.wait(HEARTBEAT_S):
            try:
                self.redis.xclaim(STREAM, GROUP, self.consumer, min_idle_time=0, message_ids=[entry_id], justid=True)
            except Exception:
                logger.warning("heartbeat failed for %s", entry_id)

    # ---------- processing ----------

    def _process(self, job_id: str, model_name: str, params: dict, input_key: str) -> dict:
        model = self.models.get(model_name) or self.models.wait_for(model_name, timeout=60)
        size, names, max_batch = model["imgsz"], model["names"], model["max_batch"]
        task = model.get("task", "detect")
        kpt_shape = tuple(model.get("kpt_shape") or (17, 3))
        sample_fps, conf = float(params.get("sample_fps", 5)), float(params.get("conf", 0.25))
        annotate, track = bool(params.get("annotate")), bool(params.get("track", True))
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / ("input" + Path(input_key).suffix)
            self.s3.download_file(self.bucket, input_key, str(src))
            cap = cv2.VideoCapture(str(src))
            if not cap.isOpened():
                raise ValueError("not a readable video")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            step = max(1, int(round(fps / sample_fps)))
            expected = (total + step - 1) // step if total else None
            with self._db() as conn:
                conn.execute("UPDATE job SET total_frames = %s WHERE id = %s", (expected, job_id))
            tracker = ByteTracker(frame_rate=fps / step) if track else None
            results_path = Path(tmp) / "detections.jsonl"
            out_path = Path(tmp) / "annotated.mp4"
            writer = self._writer(out_path, w, h, fps / step) if annotate else None
            counts: CountDict = CountDict()
            track_ids: dict = {}
            processed, idx, last_progress = 0, 0, 0.0
            pending = []   # (frame_index, frame) awaiting a batch
            with results_path.open("w") as results:
                def flush():
                    nonlocal processed
                    if not pending:
                        return
                    lbs = [letterbox(f, size) for _, f in pending]
                    outs = infer_outputs(self.triton, model["serve_name"], 1, np.stack([lb for lb, _ in lbs]),
                                         model_outputs(model), timeout_s=30.0)
                    for k, ((fi, frame), (_, meta), o) in enumerate(zip(pending, lbs, outs[0])):
                        low = min(conf, 0.1) if tracker else conf
                        if task == "pose":
                            dets = decode_pose(o, meta, names, kpt_shape, conf=low)
                        elif task == "segment":
                            # One decode; masks only for the detections that are reported (>= conf).
                            dets = decode_segment(o, outs[1][k], meta, names, size, conf=low, mask_conf=conf)
                        else:
                            dets = decode_detections(o, meta, names, conf=low)
                        tids = {}
                        if tracker:
                            b = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32).reshape(-1, 4)
                            for di, tid in tracker.update(b, np.array([d.confidence for d in dets]),
                                                          np.array([d.class_id for d in dets])):
                                tids[di] = tid
                        kept = []
                        for di, d in enumerate(dets):
                            if d.confidence < conf:
                                continue
                            rec = d.to_dict()
                            if di in tids:
                                rec["track_id"] = tids[di]
                                track_ids.setdefault(d.class_name, set()).add(tids[di])
                            counts[d.class_name] += 1
                            kept.append(rec)
                        results.write(json.dumps({"frame": fi, "time_s": round(fi / fps, 3), "detections": kept}) + "\n")
                        if writer is not None:
                            self._draw(frame, kept)
                            writer.stdin.write(frame.tobytes())
                    processed += len(pending)
                    M_FRAMES.inc(len(pending))
                    pending.clear()

                while True:
                    ok = cap.grab()
                    if not ok:
                        break
                    if idx % step == 0:
                        ok, frame = cap.retrieve()
                        if ok:
                            pending.append((idx, frame))
                            if len(pending) >= max_batch:
                                flush()
                    idx += 1
                    if time.monotonic() - last_progress > 1.0:
                        self._progress(job_id, processed)
                        last_progress = time.monotonic()
                flush()
            cap.release()
            output_key = None
            if writer is not None:
                writer.stdin.close()
                if writer.wait(timeout=600) != 0:
                    raise RuntimeError(f"ffmpeg failed: {writer.stderr.read()[-500:]}")
                output_key = f"jobs/{job_id}/annotated.mp4"
                self.s3.upload_file(str(out_path), self.bucket, output_key, ExtraArgs={"ContentType": "video/mp4"})
            result_key = f"jobs/{job_id}/detections.jsonl"
            self.s3.upload_file(str(results_path), self.bucket, result_key, ExtraArgs={"ContentType": "application/x-ndjson"})
            self._progress(job_id, processed)
        summary = {"frames_processed": processed, "video_fps": round(fps, 3), "sample_every_n_frames": step,
                   "width": w, "height": h, "model": model_name, "model_version": model["version"], "task": task,
                   "detections_by_class": dict(counts),
                   "unique_tracks_by_class": {k: len(v) for k, v in track_ids.items()} if track else None}
        return {"summary": summary, "result_key": result_key, "output_key": output_key}

    def _progress(self, job_id: str, processed: int) -> None:
        with self._db() as conn:
            status = conn.execute("UPDATE job SET processed_frames = %s WHERE id = %s RETURNING status",
                                  (processed, job_id)).fetchone()
        self._publish(job_id)
        if status and status[0] == "cancelled":
            raise JobCancelled()

    @staticmethod
    def _writer(path: Path, w: int, h: int, fps: float):
        return subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    @staticmethod
    def _draw(frame: np.ndarray, dets: list) -> None:
        polys = [(d, np.array(d["polygon"], dtype=np.int32)) for d in dets if len(d.get("polygon") or []) >= 3]
        if polys:
            overlay = frame.copy()
            for d, pts in polys:
                color = PALETTE[(d.get("track_id") or d["class_id"]) % len(PALETTE)]
                cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)
            for d, pts in polys:
                color = PALETTE[(d.get("track_id") or d["class_id"]) % len(PALETTE)]
                cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
        for d in dets:
            kp = d.get("keypoints")
            if not kp:
                continue
            color = PALETTE[(d.get("track_id") or d["class_id"]) % len(PALETTE)]
            pts = [(int(x), int(y)) if c >= KPT_DRAW_CONF else None for x, y, c in kp]
            if len(pts) == 17:
                for a, b in SKELETON:
                    if pts[a] and pts[b]:
                        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
            for p in pts:
                if p:
                    cv2.circle(frame, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
        for d in dets:
            b = d["bbox"]
            color = PALETTE[(d.get("track_id") or d["class_id"]) % len(PALETTE)]
            p1, p2 = (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"]))
            cv2.rectangle(frame, p1, p2, color, 2)
            label = f"{d['class_name']} {d['confidence']:.2f}" + (f" #{d['track_id']}" if "track_id" in d else "")
            cv2.putText(frame, label, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    start_http_server(int(os.environ.get("METRICS_PORT", "9100")))
    w = Worker()
    signal.signal(signal.SIGTERM, lambda *_: w.stop.set())
    signal.signal(signal.SIGINT, lambda *_: w.stop.set())
    w.run()


if __name__ == "__main__":
    main()
