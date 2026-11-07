# YOLO Inference Platform

A multi-model YOLO inference platform for one NVIDIA GPU server. **Triton + TensorRT**
serve the models, **FastAPI** is the public API, and **camera shards** decode RTSP streams on the
GPU (NVDEC) and share the one Triton. It handles images, video files and live cameras, with
gated model rollouts and Prometheus/Grafana monitoring.

```
 clients ──HTTPS──▶ nginx ──▶ api (FastAPI, 2 workers) ──gRPC──▶ triton (TensorRT FP16, dynamic batching)
                               │  ▲                                   ▲
                    Postgres ◀─┘  │ Redis (leases, streams, rate limits, active versions)
                                  │                                   │ system shared memory
 RTSP cameras ──▶ camera-shard ×N (PyAV demux → NVDEC → CUDA letterbox) ┘
                    │  ByteTrack per camera → rules → events ──▶ Redis Stream ──▶ event-processor ──▶ Postgres + WebSocket
 video uploads ──▶ MinIO ──▶ Redis Stream jobs:video ──▶ video-worker ──▶ results + annotated MP4 in MinIO
 Prometheus ◀── api, triton, shards, workers, DCGM exporter ──▶ Grafana
```

## Models

| Triton model | Weights | Task | Input | Notes |
|---|---|---|---|---|
| `yolo26n_detection` | yolo26n.pt | detect (80 COCO classes) | 640 | default camera model |
| `people_detection` | people.pt | detect (person) | 640 | |
| `fire_detection` | fire.pt | detect (fire, smoke, class_2) | 640 | the weights carry no names; 0/1 follow fire&smoke |
| `ppe_detection` | ppe_yolov8m.pt | detect (14 PPE classes) | 640 | YOLOv8 head, NMS embedded at export |
| `yolo26n_cls` | yolo26n-cls.pt | classify (ImageNet 1000) | 224 | for crops: an ImageNet classifier, not a person-attribute model |
| `gender_detection` | gender.pt | detect (female, male, child) | 640 | InsightEye's production gender model |
| `gender0_detection` | gender0.pt | detect (female, male) | 640 | InsightEye's older 2-class model; no names in the weights, InsightEye counts 0 = female, 1 = male |
| `yolo26l_detection` | yolo26l.pt | detect (80 COCO classes) | 640 | large: more accurate, much more GPU time per frame than yolo26n |
| `yolo26m_pose` | yolo26m-pose.pt | pose (person + 17 COCO keypoints) | 640 | camera model for `shoplifting` |
| `yolo26n_seg` | yolo26n-seg.pt | segment (80 COCO classes) | 640 | masks returned as polygons |
| `yolo11n_seg` | yolo11n-seg.pt | segment (80 COCO classes) | 640 | YOLO11 head, NMS embedded at export |
| `shoplifting_seq` | shoplifting.pt | sequence (Normal, Shoplifting) | 120×17×12 FP32 | InsightEye's Conv1d → BiGRU → attention net over one person's pose window; FP32 engine |

The last seven come from InsightEye (`/home/ubuntu/insighteye/models`). Its fire.pt, people.pt and
yolo26n.pt are the same files as `fire_detection`, `people_detection` and `yolo26n_detection`.

Every box model has an end-to-end output `[N, 300, 6 + extras]` (x1 y1 x2 y2 conf class, NMS in the
engine): pose adds 17 × (x, y, conf), segmentation adds 32 mask coefficients plus a second output
with the mask prototypes `[N, 32, 160, 160]`. Each image engine takes **UINT8 NHWC RGB** input:
normalisation and the transpose run on the GPU, inside the engine. The sequence engine takes FP32
feature windows (`shared/yolo_platform_shared/behavior.py`) and z-scores and softmaxes inside the engine.
All twelve engines together take about 3 GB of VRAM in Triton (measured on the T4).

The FP16 engines' embedded NMS occasionally keeps a near-identical second box of the same
class (IoU above 0.9; the FP32 graph never does). Decoding therefore suppresses boxes again,
class-aware at IoU 0.7 (Ultralytics' default), and the rollout gate measures the same rows.

## Quick start

```bash
cp .env.example .env                  # then set the secrets (ADMIN_API_KEY: see the comment)
# 1. Models: weights -> ONNX -> TensorRT plans -> Triton configs -> parity check
cp <weights> inference/weights/       # names as in inference/models.yaml
docker build -t yolo-platform-export:dev inference/export
docker run --rm -u $(id -u):$(id -g) -v $PWD/inference:/work yolo-platform-export:dev python export/export_onnx.py
for m in $(ls inference/build); do docker run --rm -u $(id -u):$(id -g) -v $PWD/inference:/work yolo-platform-export:dev \
  python export/wrap_uint8.py build/$m/model.fp32.onnx build/$m/model.onnx; done
inference/export/build_plan.sh 1      # trtexec from the pinned Triton image, FP16
docker run --rm -u $(id -u):$(id -g) -v $PWD/inference:/work yolo-platform-export:dev python export/gen_config.py
# 2. Stack
docker compose up -d --build
# 3. Golden files for the rollout gate (needs Triton up; see "Parity" below)
```

On its first start the API loads each model's newest version and gates it against `golden.npz`.
Only a version that passes is marked active.

Ports: nginx `8088` (HTTP, redirects) and `8443` (HTTPS), Grafana `3001`, Prometheus `127.0.0.1:9091`,
MinIO console `127.0.0.1:9011`, test RTSP rig `8558`. All of them are set in `.env`. Triton, Redis,
Postgres and the API have no host ports.

TLS: by default, a self-signed certificate is created in `deploy/nginx/certs/` on first start.
To use a real one, put your `tls.crt` and `tls.key` there instead.

## API

Authenticate with the `X-API-Key: yp_<prefix>_<secret>` header. For WebSockets, `?api_key=` also works.
Swagger is at `/docs`. `/health/ready` and `/v1/models/{name}` also name the Triton model that serves each version (`<name>__v<version>`).

| Method | Path | Scope | |
|---|---|---|---|
| GET | `/health/live`, `/health/ready` | – | ready = Postgres, Redis, Triton and every active model |
| GET | `/v1/models`, `/v1/models/{name}` | infer | versions, status, gate reports, classes |
| POST | `/v1/infer/images` | infer | `file`, `model`, `conf`, `classes`, `max_det`, `classify_model`, `classify_classes`, `top_k` |
| POST | `/v1/infer/batch` | infer | `files` (≤ 32), same options; one rate-limit token per image |
| POST | `/v1/infer/sequences` | infer | JSON: `model`, `keypoints` (≤ 64 windows of ≤ 120 × 17 × [x, y], normalised) or `features`; one token per window |
| POST | `/v1/infer/videos` | jobs | upload (≤ 2 GB), `model`, `sample_fps`, `conf`, `annotate`, `track` → `202 {job_id}` |
| GET | `/v1/jobs`, `/v1/jobs/{id}` | jobs | progress, summary, `result_url` (JSONL), `output_url` (MP4) |
| POST | `/v1/jobs/{id}/cancel` | jobs | |
| WS | `/v1/jobs/{id}/stream` | jobs | progress until done |
| CRUD | `/v1/cameras`, `/v1/cameras/{id}` | cameras | rules and triggered models in `config` |
| POST | `/v1/cameras/{id}/start`, `/stop` | cameras | |
| GET | `/v1/cameras/{id}/events` | cameras | newest first, `before_id` pages |
| WS | `/v1/cameras/{id}/stream` | cameras | live events + status every 2 s |
| POST | `/v1/models/{name}/versions/{v}` | admin | upload `plan` + `golden` (+ `meta`) |
| POST | `/v1/models/{name}/versions/{v}/deploy` | admin | load next to the active version → gate → switch → drain the old one |
| POST | `/v1/models/{name}/rollback` | admin | back to the previously active version (gated again) |
| * | `/v1/admin/tenants`, `/v1/admin/keys` | admin | tenants (model allowlist, camera limit), keys (scopes, rate/min) |

```bash
curl -k -H "X-API-Key: $KEY" -F file=@street.jpg -F classify_model=yolo26n_cls -F classify_classes=0 \
  https://localhost:8443/v1/infer/images
```

Response (abridged):

```json
{"request_id": "req_…", "model": "yolo26n_detection", "model_version": "1", "task": "detect",
 "timing": {"preprocess_ms": 4.3, "inference_ms": 23.9, "postprocess_ms": 0.3, "total_ms": 40.1},
 "images": [{"index": 0, "width": 640, "height": 480, "detections": [
   {"class_id": 0, "class_name": "person", "confidence": 0.87, "bbox": {"x1": 428.5, "y1": 144.5, "x2": 520.0, "y2": 344.5},
    "classification": [{"class_id": 439, "class_name": "bearskin", "confidence": 0.12}]}]}]}
```

Pose models add `"keypoints": [[x, y, conf], …]` (17, image pixels) to each detection, and
segmentation models add `"polygon": [[x, y], …]`: the outline of the largest region of the
instance mask, in image pixels. The mask is computed as Ultralytics does (prototype logits
upsampled bilinearly to the input size, cropped to the box, > 0).

Sequences: one window is one person's pose over time, oldest sample first, keypoints normalised
to the frame (`null` for a sample where the person was not seen). The server builds the
features exactly as cameras do and left-pads windows shorter than 120 samples.

```bash
curl -k -H "X-API-Key: $KEY" -H 'Content-Type: application/json' https://localhost:8443/v1/infer/sequences \
  -d '{"model": "shoplifting_seq", "keypoints": [[ [[0.51,0.32], …17 points…], null, … ]]}'
# {"task": "sequence", "results": [{"index": 0, "class_name": "Normal", "confidence": 0.97,
#   "probabilities": {"Normal": 0.97, "Shoplifting": 0.03}}], …}
```

Limits:
- 10 MB and 64 MP per image, 32 images per batch.
- The pixel check reads the image header before decoding, so decompression bombs are rejected.
- `503` with `Retry-After` when 64 requests per API process are already in flight, or when Triton's queue rejects the request.
- `429` with `Retry-After` past the key's per-minute budget.

### Cameras

```json
{"name": "gate-3", "url": "rtsp://user:pass@10.0.0.13/stream2", "target_fps": 5, "model": "yolo26n_detection",
 "enabled": true,
 "config": {
   "rules": [
     {"name": "loiter", "type": "zone_dwell", "zone": [[0.1,0.5],[0.6,0.5],[0.6,1],[0.1,1]], "dwell_s": 20, "cooldown_s": 60},
     {"name": "entry", "type": "line_cross", "line": [[0.5,0],[0.5,1]], "classes": ["person","car"]},
     {"name": "crowd", "type": "count_above", "count": 10},
     {"name": "person", "type": "presence", "classify": {"model": "yolo26n_cls", "top_k": 1}}],
   "triggered": [
     {"model": "fire_detection", "every_s": 2, "min_confidence": 0.5},
     {"model": "ppe_detection", "on_rule": "loiter", "classes": ["NO-Hardhat"], "min_confidence": 0.5}]}}
```

- **Geometry:** zones and lines are normalised to 0..1, so a rule keeps working on the main stream or the substream. Objects are anchored at their bottom centre.
- **Primary detector:** runs on every sampled frame. Its output goes through one ByteTrack per camera and then the rules, each rule with its own cooldown.
- **Triggered models:** run only when due (`every_s`) or when a rule fires (`on_rule`), so the GPU is not spent on every model for every frame.
- **Config changes:** a change to `config` or `target_fps` is hot-swapped without reconnecting. A change to `url` or `model` restarts that camera's reader.
- **Camera models:** a detection or a pose model at 640 px. Segmentation is offered for images and video jobs only: per frame its mask prototypes are larger than the frame.

**Shoplifting** (pose camera model only), ported from InsightEye through the standalone shoplifting app:

```json
{"name": "aisle-4", "url": "rtsp://…", "target_fps": 5, "model": "yolo26m_pose",
 "config": {"shoplifting": {"threshold": 0.5, "sample_fps": 3, "smooth_window": 3, "min_window_samples": 30,
                            "min_person_height_px": 60, "track_cooldown_s": 300, "camera_cooldown_s": 30}}}
```

- Every tracked person has their own window of 120 pose samples (40 s at `sample_fps` 3, as trained); a lost person gets empty rows, and a new track id that appears within 5 s of someone vanishing, with a similar height, takes over that person's window.
- Once a window has `min_window_samples` real samples, each new sample is judged by `shoplifting_seq` on a separate thread, batched across the shard's cameras. The probability is averaged over the last `smooth_window` judgements.
- At `threshold` the shard emits an event of type `shoplifting` (`data`: probability, peak, window samples), at most once per person per `track_cooldown_s` and once per camera per `camera_cooldown_s`. InsightEye ran its threshold at 0.15; 0.5 is the class boundary.
- `rules` and `triggered` models work on a pose camera as on a detector (on the person boxes).

**Use the camera's substream** (e.g. 640×360 or D1) wherever its resolution is enough for detection.
The detector sees 640×640 either way, and NVDEC and network costs grow with the stream's pixel count.

## How camera processing works

- **Shards and decode:**
  - Each `camera-shard` replica is one process with one CUDA context.
  - For each camera it owns, the shard runs a reader thread: PyAV demuxes (RTSP over TCP, 10 s socket timeouts) and NVDEC decodes.
  - When a frame is due (by `target_fps`, on the wall clock), a CUDA kernel letterboxes it on the GPU. Only the 640×640 input is copied to host memory, straight into a Triton **system shared-memory** batch slot.
- **Latest frame wins:** a sampled frame that the batcher hasn't taken yet is overwritten by the next one and counted as *dropped*. A slow GPU therefore shows up as dropped frames, never as growing latency.
- **Batching:** two inference pipelines per shard batch the fresh frames of every camera that uses the same model, and Triton's dynamic batcher merges requests across shards.
- **Ownership:** cameras are assigned through Redis leases (`workers/scheduler.py`):
  - each shard claims its fair share, and a lease is taken with SET NX and renewed only by its holder;
  - when a shard dies its leases expire after 15 s, and the survivors take its cameras (measured at about 20 s end to end).
- **Reconnects:** exponential backoff (1 s up to 30 s), staggered stream opens, and a supervisor that replaces a reader that has stopped delivering frames.
- **Fallback:** `DECODE_BACKEND=cpu` switches to OpenCV decoding with a drain thread, as a fallback and as a baseline.

## Video jobs

- **Path:** upload → MinIO → `job` row → Redis Stream `jobs:video` (consumer group) → `video-worker`.
- **Heartbeat:** a running job touches its stream entry every 20 s.
- **Crashed workers:** a worker that dies leaves the entry pending, and another worker reclaims it with `XAUTOCLAIM` after 120 s.
- **Retries:** after 3 attempts the job is marked failed and copied to `jobs:video:dead`.
- **Models:** detection, pose and segmentation models. Pose results carry keypoints and segmentation results carry polygons; the annotated video draws skeletons and mask outlines.
- **Results:** `detections.jsonl` (one line per sampled frame, with track ids), a summary (counts per class, unique tracks), and an annotated H.264 MP4 when `annotate=true`.
- **Links:** the result links are presigned, and returned as `/storage/...` paths that nginx serves.

## Model lifecycle

1. **Source:** the `.pt` weights in `inference/weights/` are the source of truth.
2. **Export:** `export_onnx.py` inspects each model (task, head, classes, tensor I/O → `build/<name>/meta.json`) and exports it to ONNX. Output shapes come from an ONNX Runtime dry run, since some exported dims are symbolic. Non-YOLO sequence networks are defined in `export/sequence_models.py` and chosen by `builder` in `models.yaml`.
3. **uint8 input:** `wrap_uint8.py` adds the UINT8 NHWC input (sequence models are copied unchanged).
4. **Engine:** `build_plan.sh <version>` runs `trtexec` from the **same Triton image** that serves the engines, because TensorRT plans only load on the TensorRT version and GPU that built them. Output is FP16 (FP32 where `models.yaml` says `precision: fp32`) with a batch profile of 1 / opt / max.
5. **Triton config:** `gen_config.py` writes `config.pbtxt` from the real tensor I/O, with dynamic batching, a bounded queue that rejects on timeout, one GPU instance, and warm-up.
6. **Parity:** `validation/parity.py --write-golden` compares the engine with the `.pt` on frames sampled from the clips:
   - detectors: recall and precision of confident boxes ≥ 0.95, mean IoU ≥ 0.90, mean confidence difference ≤ 0.05;
   - pose: as detectors, plus the mean distance of matched keypoints (reference conf ≥ 0.5) ≤ 3 px at 640;
   - segmentation: as detectors, plus the mean mask IoU of matched boxes ≥ 0.85 (masks at input size, sampled every 4 px);
   - classifiers: top-1 agreement ≥ 0.95;
   - sequence: top-1 agreement ≥ 0.95 and mean probability difference ≤ 0.02. The windows are built from the clips with yolo26m-pose + ByteTrack at 3 fps, plus shuffled and random windows, since the clips show no shoplifting.

   It then saves `golden.npz` (the inputs plus the `.pt` outputs), so the API can re-run the same gate without torch.
7. **Rollout:** `POST /v1/models/{name}/versions/{v}` uploads the version, then `.../deploy`:
   - the API generates the Triton model `<name>__v<v>` in the serving repository (`inference/serving/`: the config is derived from the source config, the plan is a symlink) and loads it next to the serving one;
   - the gate runs against v's golden file and its sha256 is checked;
   - traffic switches to v (Postgres, then the Redis mirror the workers read);
   - the previous model is unloaded after a 10 s drain;
   - a failing version is marked `failed` and unloaded, so it never serves;
   - `rollback` re-activates the previous version through the same gate.
8. **Watchdog:** a model that is serving is **never reloaded in place**. With Triton 25.10, reloading a detector (embedded NMS) while one of its versions was kept made the next request to that version segfault the server (reproduced, see the commit). The API's watchdog reloads active models after a Triton restart (explicit mode comes back empty) and unloads strays.

Parity on 2026-11-04 (FP16 engines vs FP32 `.pt`; precision before duplicate suppression):

| Model | Recall | Precision | Mean IoU | Mean conf diff |
|---|---|---|---|---|
| yolo26n_detection | 1.000 | 0.967 | 0.993 | 0.001 |
| people_detection | 1.000 | 0.999 | 0.989 | 0.001 |
| fire_detection | 1.000 | 1.000 | 0.992 | 0.003 |
| ppe_detection | 1.000 | 1.000 | 0.994 | 0.001 |
| yolo26n_cls | top-1 agreement 1.000 | | | 0.005 |

Parity on 2026-11-06 for the InsightEye models (120 frames, or 48 pose windows for the sequence model):

| Model | Recall | Precision | Mean IoU | Mean conf diff | Task metric |
|---|---|---|---|---|---|
| gender_detection | 1.000 | 1.000 | 0.992 | 0.002 | |
| gender0_detection | 1.000 | 1.000 | 0.992 | 0.001 | |
| yolo26l_detection | 1.000 | 0.972 | 0.994 | 0.002 | |
| yolo26m_pose | 1.000 | 1.000 | 0.994 | 0.002 | keypoint error 0.12 px |
| yolo26n_seg | 1.000 | 0.955 | 0.993 | 0.002 | mask IoU 0.995 |
| yolo11n_seg | 1.000 | 0.963 | 0.993 | 0.002 | mask IoU 0.996 |
| shoplifting_seq (FP32) | top-1 agreement 1.000 | | | < 0.0001 | 29 Normal / 19 Shoplifting windows |

With duplicate suppression, yolo26n_seg and yolo26n_detection both reach precision 1.000 on their golden files.

INT8 is deliberately not enabled. It needs representative calibration data, and its accuracy has to be compared against FP16 before it is used.

## Monitoring

- **Prometheus** scrapes the API, Triton, every shard and worker replica (through Docker DNS), and the DCGM exporter.
- **Grafana** (`:3001`) provisions the *YOLO Inference Platform* dashboard, which is generated by `monitoring/gen_dashboard.py`. It covers:
  - GPU and NVDEC utilisation, VRAM, temperature and power;
  - Triton throughput, batch size, queue and compute time;
  - API p50/p95/p99;
  - per-camera fps and drops, decode-to-result latency;
  - events, triggered models and jobs.
- **Alerts** are in `monitoring/alerts.yml`: Triton or API down, p95 latency, Triton queue and failures, shard down, frame drops, no frames, storage failures, GPU memory or temperature, NVDEC saturation.

## Operations

- **Scale shards:** `CAMERA_SHARDS=4 docker compose up -d camera-shard`. Leases rebalance on their own. One shard per vCPU is a good start.
- **Backups:** the `backup` service runs `pg_dump` nightly at 02:30 into `./backups` and keeps 14 days. Restore with `gunzip -c backups/<file> | docker compose exec -T postgres psql -U inference -d inference`. Redis runs with AOF (`appendfsync everysec`, `noeviction`).
- **Test RTSP rig:**
  - `tools/gen_substreams.sh` produces `videos/sub/clipN.mp4` (≈640×360) and `videos/main/clipN.mp4` (1080p);
  - start it with `docker compose --profile testsrc up -d mediamtx`, then use `rtsp://mediamtx:8554/sub0..5` or `main0..5`;
  - stop it when done, because looping clips keep cameras and events busy.
- **Tests:**
  - `tools/e2e.sh` builds a scratch stack (`-p yp-test`, no host ports, a copy of the model repository), then runs unit tests, the integration suite through nginx, and three failure drills, and finally removes the stack. The drills:
    - kill a shard: its cameras move to the survivor, measured at 19 s;
    - kill the video worker mid-job: the job is reclaimed and completes on attempt 2;
    - restart Triton: the API is ready again in 13 s and the cameras are inferring again in 25 s;
  - unit tests alone: see `.github/workflows/ci.yml`.
- **Benchmark:** `tools/bench.py` (see *Capacity* below).
- **CI/CD:**
  - `ci.yml` covers compose config, lint, the dashboard drift check, image builds and unit tests;
  - `deploy.yml` runs on `main` only when the repository variable `DEPLOY_ENABLED=true`. It deploys over SSH, rebuilds only the images whose inputs changed, and rolls back to the previous images if the stack is not healthy.

## Capacity

These figures were measured on 2026-11-04 with `tools/bench.py`.
- **Hardware:** 1× Tesla T4 (16 GB), **4 vCPU**, 15 GB RAM.
- **Setup:** 4 shards × 2 pipelines, yolo26n on every frame at **5 fps per camera**, one presence rule per camera.
- **Streams:** H.264 test streams served by the mediamtx rig on the same host. The rig itself used ≈17% of one core, which is load a production setup would not have.
- **Pass criterion:** "Keeps up" means every camera got ≥ 90% of its target fps and < 5% of sampled frames were dropped as stale.

**Substreams (≈640×360), NVDEC decode (default)**

| Cameras | Inferred fps | Min fps/cam | Dropped | p95 latency | Avg Triton batch | GPU % | NVDEC % | Host CPU % | Keeps up |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 50.0 | 5.00 | 0% | 39 ms | 1.4 | 37 | 6 | 25 | yes |
| 25 | 125.0 | 5.00 | 0% | 42 ms | 2.3 | 53 | 9 | 41 | yes |
| 50 | 250.0 | 4.98 | 0% | 71 ms | 3.9 | 67 | 21 | 59 | yes |
| 75 | 374.9 | 4.98 | 0% | 95 ms | 6.8 | 81 | 32 | 80 | yes |
| **85** | **424.9** | **4.96** | **0%** | **195 ms** | 11.5 | 84 | 37 | 86 | **yes: ceiling** |
| 90 | 420.2 | 4.42 | 6.4% | 468 ms | 11.4 | 86 | 43 | 85 | no |
| 100 | 422.7 | 4.04 | 15.5% | 481 ms | 12.8 | 82 | 45 | 88 | no |

**1080p main streams, NVDEC decode**

| Cameras | Inferred fps | Min fps/cam | Dropped | p95 latency | GPU % | NVDEC % | Host CPU % | Keeps up |
|---|---|---|---|---|---|---|---|---|
| 25 | 125.0 | 5.00 | 0% | 45 ms | 54 | 53 | 45 | yes |
| 40 | 200.1 | 5.00 | 0% | 48 ms | 75 | 91 | 68 | yes |
| 55 | 270.4 | 4.86 | 0% | 91 ms | 79 | **99** | 86 | yes, NVDEC saturated |

**Substreams, CPU decode (OpenCV, `DECODE_BACKEND=cpu`)**: 25 cameras keep up at 71% CPU. At 50 cameras the CPU is at 100%, only 191 of 250 fps get through, and 23% of frames drop.

**What this means**
- **On this box the ceiling is about 85 substream cameras at 5 fps** (≈425 detector inferences/s), or about 55 cameras on 1080p main streams, where NVDEC saturates first. 100 cameras at 5 fps does not fit. Past the ceiling the platform degrades the intended way: each camera drops stale frames and its latency stays bounded, instead of a backlog building up.
- **At the substream ceiling the GPU (≈85%) and the 4 vCPU (≈86%) run out together.** Triton's queue time climbs from 10 ms at 75 cameras to 150+ ms at 100.
- **A second engine instance for yolo26n changed nothing.** Measured at 75 and 100 cameras: 416.6 fps against 422.7.
- **Two optimisations are already in:**
  - NVDEC returns NV12, and one CUDA kernel does colour conversion plus letterbox for the sampled frames only. This took decode CPU from 0.65 to 0.22 ms per decoded frame, moved the clean ceiling from 75 cameras (borderline before: 4.6% drops, 429 ms p95) to 85, and lifted 100-camera throughput from 309 to 423 fps.
  - Frames reach Triton through shared memory.
- **To reach 100+ cameras at 5 fps with the same models:**
  - ≥ 8 vCPU, because with 4 vCPU the CPU is at 86% at 85 cameras;
  - substreams;
  - plus one of: a second T4 (a second Triton, with shards split between the two), a GPU with more headroom such as an L4, or a lower detector rate. At 3 fps the GPU side would have room for about 140 cameras.
    However, every camera is still decoded at its full stream rate whatever the detector rate is,
    so the CPU and NVDEC cost per camera stays the same. Lower the stream's own frame rate as well
    (10–12 fps on the camera substream), or add CPU.
  - Each extra triggered model adds its own GPU load. Run them `on_rule`, or with `every_s` of 1 s or more, not on every frame.
- **Not measured:** INT8 engines, and cameras whose streams are not H.264 or HEVC.

Raw results are in `artifacts/bench/*.json` and `*.md`.

## Out of scope for now

- INT8 engines (they need calibration data).
- Multi-GPU and Kubernetes. The shards and the API take one `TRITON_URL`; a second GPU node means a second Triton and splitting shards between them.
- A web console: the API, Swagger and Grafana are the UI.
- The yolo26n-cls model is an ImageNet classifier. "Classify on crop" is wired up and works, but useful crop attributes need a classifier trained for them.
