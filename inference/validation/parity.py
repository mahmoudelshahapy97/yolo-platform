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
sys.path.insert(0, str(ROOT / "export"))
from yolo_platform_shared.behavior import NUM_FRAMES, keypoints_xyn, windows_from_sequence  # noqa: E402
from yolo_platform_shared.gate import evaluate, sample_masks  # noqa: E402
from yolo_platform_shared.preprocess import classify_crop, letterbox  # noqa: E402

# Clips that contain each model's subject, so "0 of 0 matched" cannot pass.
CLIPS = {"fire_detection": ["fire1.mp4"]}
DEFAULT_CLIPS = ["cctv_video1.mp4", "cctv_video3.mp4", "cctv_video20.mp4", "normal-1.mp4", "5.mp4"]
GOLDEN_IMAGES = 32  # inputs kept in golden.npz (~13 MB at 640 px)
GOLDEN_SEQUENCES = 96  # sequence windows kept (~9 MB)


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


def triton_batched(client, model: str, version: str, inputs: np.ndarray, max_batch: int, meta: dict) -> list:
    """All of the model's outputs on ``inputs`` (config order), chunked to the engine's batch size."""
    in_name = meta.get("input", {}).get("name", "images")
    names = [o["name"] for o in meta.get("outputs", [{"name": "output0"}])]
    outs = {n: [] for n in names}
    for start in range(0, len(inputs), max_batch):
        chunk = np.ascontiguousarray(inputs[start:start + max_batch])
        inp = grpcclient.InferInput(in_name, list(chunk.shape), "UINT8" if chunk.dtype == np.uint8 else "FP32")
        inp.set_data_from_numpy(chunk)
        res = client.infer(model, [inp], model_version=version,
                           outputs=[grpcclient.InferRequestedOutput(n) for n in names])
        for n in names:
            outs[n].append(res.as_numpy(n))
    return [np.concatenate(outs[n]) for n in names]


def reference_detect(pt: YOLO, inputs: np.ndarray, size: int, task: str = "detect") -> dict:
    """The .pt's boxes (and keypoints / masks) on the letterboxed inputs, model space."""
    boxes, conf, cls, img, kpts, masks = [], [], [], [], [], []
    for n, rgb in enumerate(inputs):
        r = pt.predict(np.ascontiguousarray(rgb[..., ::-1]), imgsz=size, conf=0.25, device="cpu", verbose=False)[0]
        k = len(r.boxes)
        boxes.append(r.boxes.xyxy.numpy().reshape(-1, 4))
        conf.append(r.boxes.conf.numpy())
        cls.append(r.boxes.cls.numpy().astype(np.int32))
        img.append(np.full(k, n, dtype=np.int32))
        if task == "pose":
            kpts.append(r.keypoints.data.numpy().reshape(k, *pt.model.model[-1].kpt_shape))
        if task == "segment":
            s = size // 4
            m = r.masks.data.numpy().astype(bool) if r.masks is not None else np.zeros((0, size, size), bool)
            masks.append(sample_masks(m) if k else np.zeros((0, s, s), bool))
    out = {
        "ref_boxes": np.concatenate(boxes).astype(np.float32),
        "ref_conf": np.concatenate(conf).astype(np.float32),
        "ref_cls": np.concatenate(cls),
        "ref_img": np.concatenate(img),
    }
    if task == "pose":
        out["ref_kpts"] = np.concatenate(kpts).astype(np.float32)
    if task == "segment":
        out["ref_masks"] = np.concatenate(masks)
    return out


def sequence_inputs(videos: Path, clips: list, sample_fps: float = 3.0, max_samples: int = 150,
                    seed: int = 0) -> np.ndarray:
    """Feature windows as cameras build them: yolo26m-pose + ByteTrack at ``sample_fps`` over the clips,
    per-person windows ending at several points, plus synthetic windows so that probabilities
    across the whole range are compared (the clips show no shoplifting)."""
    pose = YOLO(str(ROOT / "weights" / "yolo26m-pose.pt"))
    windows = []
    for clip in clips:
        cap = cv2.VideoCapture(str(videos / clip))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps / sample_fps)))
        seqs: dict = {}
        fi = samples = 0
        while samples < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            fi += 1
            if fi % step:
                continue
            samples += 1
            r = pose.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=640, conf=0.25, device="cpu",
                           verbose=False)[0]
            ids = r.boxes.id.numpy().astype(int) if r.boxes.id is not None else []
            h, w = frame.shape[:2]
            for tid, kp in zip(ids, r.keypoints.data.numpy()):
                seqs.setdefault(int(tid), [None] * (samples - 1))
                seqs[int(tid)].append(keypoints_xyn(kp, w, h))
            for tid, seq in seqs.items():
                if len(seq) < samples:
                    seq.append(None)
        cap.release()
        for seq in seqs.values():
            real = sum(x is not None for x in seq)
            if real < 30:
                continue
            for end in np.linspace(30, len(seq), 4).astype(int):
                windows.append(windows_from_sequence(seq[:end]))
    rng = np.random.default_rng(seed)
    real = np.stack(windows) if windows else np.zeros((0, NUM_FRAMES, 17, 12), np.float32)
    synth = []
    for i in range(32):
        if len(real) and i % 2 == 0:
            w = real[rng.integers(len(real))].copy()
            rng.shuffle(w)                                # same poses, scrambled in time
            synth.append(w + rng.normal(0, 0.02, w.shape).astype(np.float32))
        else:
            seq = [rng.uniform(0.2, 0.8, (17, 2)).astype(np.float32) for _ in range(NUM_FRAMES)]
            synth.append(windows_from_sequence(seq))
    return np.concatenate([real, np.stack(synth)]).astype(np.float32)


def reference_sequence(meta: dict, inputs: np.ndarray) -> dict:
    import torch

    from sequence_models import BUILDERS, ExportWrapper

    loader = BUILDERS[meta.get("builder", "shoplifting")][0]
    net = ExportWrapper(loader(str(ROOT / "weights" / meta["weights"]))).eval()
    with torch.no_grad():
        probs = net(torch.from_numpy(inputs)).numpy()
    return {"ref_top1": probs.argmax(1).astype(np.int64), "ref_top1_prob": probs.max(1).astype(np.float32)}


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
    if task in ("classify", "sequence"):
        out["ref_top1"] = golden["ref_top1"][picks]
        out["ref_top1_prob"] = golden["ref_top1_prob"][picks]
        return out
    renumber = {int(old): new for new, old in enumerate(picks)}
    sel = np.isin(golden["ref_img"], picks)
    for k in ("ref_boxes", "ref_conf", "ref_cls", "ref_kpts", "ref_masks"):
        if k in golden:
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
        clips = CLIPS.get(name, DEFAULT_CLIPS)
        if task == "sequence":
            build_meta = json.loads((ROOT / "build" / name / "meta.json").read_text())
            inputs = sequence_inputs(Path(args.videos), clips[:3])
            golden = {"inputs": inputs, **reference_sequence(build_meta, inputs)}
        else:
            frames = [f for c in clips for f in sample_frames(Path(args.videos) / c, args.frames)]
            prep = (lambda f: classify_crop(f, size)) if task == "classify" else (lambda f: letterbox(f, size)[0])
            inputs = np.stack([prep(f) for f in frames])
            pt = YOLO(str(ROOT / "weights" / meta["weights"]))
            golden = {"inputs": inputs, **(reference_classify(pt, inputs, size) if task == "classify"
                                           else reference_detect(pt, inputs, size, task))}
        outputs = triton_batched(client, name, args.version, inputs, meta["max_batch"], meta)
        result = evaluate(task, golden, outputs, meta)
        result.update(seconds=round(time.time() - started, 1), version=args.version, clips=clips)
        report[name] = result
        ok &= result["passed"]
        print(json.dumps({name: {k: v for k, v in result.items() if k != "thresholds"}}), flush=True)
        if args.write_golden and result["passed"]:
            target = ROOT / "model_repository" / name / args.version / "golden.npz"
            keep = GOLDEN_SEQUENCES if task == "sequence" else GOLDEN_IMAGES
            np.savez_compressed(target, **subset(golden, keep, task))
    (ROOT / "build" / f"parity-v{args.version}.json").write_text(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
