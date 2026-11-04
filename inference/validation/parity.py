"""Parity check: a TensorRT engine served by Triton must agree with its .pt.

For each model, frames sampled from the test clips are letterboxed (our code,
the same the API and workers use) and go through
  (a) Ultralytics on the original .pt (FP32, CPU), and
  (b) Triton, the FP16 engine of the given version.
Both see the identical letterboxed pixels, so this checks the engine, the
UINT8 wrapper and the decoding. That our letterbox equals Ultralytics' own is
covered separately by tests/test_preprocess.py.

The comparison (shared/yolo_platform_shared/gate.py) is the same one the API's
rollout gate runs. With --write-golden, the inputs and the .pt reference outputs
are saved to model_repository/<name>/<version>/golden.npz so the API can re-run
the gate later without torch.

Exit code 1 if any model fails. Writes build/parity-v<version>.json.

    python validation/parity.py --triton triton:8001 --videos /videos [--only m ...] [--version 1] [--write-golden]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "shared"))
from yolo_platform_shared.gate import evaluate_classify, evaluate_detect  # noqa: E402
from yolo_platform_shared.preprocess import classify_crop, letterbox  # noqa: E402

# Clips that contain each model's subject, so "0 of 0 matched" cannot pass.
CLIPS = {"fire_detection": ["fire1.mp4"]}
DEFAULT_CLIPS = ["cctv_video1.mp4", "cctv_video3.mp4", "cctv_video20.mp4", "normal-1.mp4", "5.mp4"]
GOLDEN_IMAGES = 32  # inputs kept in golden.npz (~13 MB at 640 px)


def sample_frames(video: Path, count: int) -> list:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for idx in np.linspace(0, total - 1, count).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"no frames read from {video}")
    return frames


def triton_batched(client, model: str, version: str, inputs: np.ndarray, max_batch: int) -> np.ndarray:
    outs = []
    for start in range(0, len(inputs), max_batch):
        chunk = inputs[start:start + max_batch]
        inp = grpcclient.InferInput("images", list(chunk.shape), "UINT8")
        inp.set_data_from_numpy(chunk)
        res = client.infer(model, [inp], model_version=version, outputs=[grpcclient.InferRequestedOutput("output0")])
        outs.append(res.as_numpy("output0"))
    return np.concatenate(outs)


def reference_detect(pt: YOLO, inputs: np.ndarray, size: int) -> dict:
    boxes, conf, cls, img = [], [], [], []
    for n, rgb in enumerate(inputs):
        r = pt.predict(np.ascontiguousarray(rgb[..., ::-1]), imgsz=size, conf=0.25, device="cpu", verbose=False)[0]
        k = len(r.boxes)
        boxes.append(r.boxes.xyxy.numpy().reshape(-1, 4))
        conf.append(r.boxes.conf.numpy())
        cls.append(r.boxes.cls.numpy().astype(np.int32))
        img.append(np.full(k, n, dtype=np.int32))
    return {
        "ref_boxes": np.concatenate(boxes).astype(np.float32),
        "ref_conf": np.concatenate(conf).astype(np.float32),
        "ref_cls": np.concatenate(cls),
        "ref_img": np.concatenate(img),
    }


def reference_classify(pt: YOLO, inputs: np.ndarray, size: int) -> dict:
    top1, prob = [], []
    for rgb in inputs:
        r = pt.predict(np.ascontiguousarray(rgb[..., ::-1]), imgsz=size, device="cpu", verbose=False)[0]
        top1.append(int(r.probs.top1))
        prob.append(float(r.probs.top1conf))
    return {"ref_top1": np.array(top1, dtype=np.int64), "ref_top1_prob": np.array(prob, dtype=np.float32)}


def subset(golden: dict, keep: int, task: str) -> dict:
    """``keep`` inputs spread evenly over all clips, with their reference rows renumbered."""
    total = len(golden["inputs"])
    picks = np.unique(np.linspace(0, total - 1, min(keep, total)).astype(int))
    out = {"inputs": golden["inputs"][picks]}
    if task == "classify":
        out["ref_top1"] = golden["ref_top1"][picks]
        out["ref_top1_prob"] = golden["ref_top1_prob"][picks]
        return out
    renumber = {int(old): new for new, old in enumerate(picks)}
    sel = np.isin(golden["ref_img"], picks)
    for k in ("ref_boxes", "ref_conf", "ref_cls"):
        out[k] = golden[k][sel]
    out["ref_img"] = np.array([renumber[int(i)] for i in golden["ref_img"][sel]], dtype=np.int32)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triton", default="triton:8001")
    parser.add_argument("--videos", default=str(ROOT.parent / "videos"))
    parser.add_argument("--frames", type=int, default=24, help="frames per clip")
    parser.add_argument("--version", default="1")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--write-golden", action="store_true")
    args = parser.parse_args()

    client = grpcclient.InferenceServerClient(args.triton)
    report, ok = {}, True
    for meta_path in sorted((ROOT / "model_repository").glob("*/model_meta.json")):
        meta = json.loads(meta_path.read_text())
        name, size, task = meta["name"], meta["imgsz"], meta["task"]
        if args.only and name not in args.only:
            continue
        started = time.time()
        frames = [f for c in CLIPS.get(name, DEFAULT_CLIPS) for f in sample_frames(Path(args.videos) / c, args.frames)]
        prep = (lambda f: classify_crop(f, size)) if task == "classify" else (lambda f: letterbox(f, size)[0])
        inputs = np.stack([prep(f) for f in frames])
        pt = YOLO(str(ROOT / "weights" / meta["weights"]))
        golden = {"inputs": inputs, **(reference_classify(pt, inputs, size) if task == "classify" else reference_detect(pt, inputs, size))}
        outputs = triton_batched(client, name, args.version, inputs, meta["max_batch"])
        result = (evaluate_classify if task == "classify" else evaluate_detect)(golden, outputs)
        result.update(seconds=round(time.time() - started, 1), version=args.version, clips=CLIPS.get(name, DEFAULT_CLIPS))
        report[name] = result
        ok &= result["passed"]
        print(json.dumps({name: {k: v for k, v in result.items() if k != "thresholds"}}), flush=True)
        if args.write_golden and result["passed"]:
            target = ROOT / "model_repository" / name / args.version / "golden.npz"
            np.savez_compressed(target, **subset(golden, GOLDEN_IMAGES, task))
    (ROOT / "build" / f"parity-v{args.version}.json").write_text(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
