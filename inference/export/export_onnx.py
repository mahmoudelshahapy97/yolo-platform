"""Inspect each model in models.yaml and export it to ONNX.

For every model this writes build/<name>/model.fp32.onnx plus build/<name>/meta.json.
meta.json records what the inspection found: task, class names, architecture,
whether NMS is embedded, and the graph's input and output tensors with concrete
shapes (from an ONNX Runtime dry run, because some exported dims are symbolic).
Everything downstream (the Triton config, the API adapters, the parity check)
reads meta.json and never guesses at tensor shapes.

Tasks:
  detect    output0 [N, 300, 6]            x1 y1 x2 y2 conf cls (NMS embedded)
  pose      output0 [N, 300, 6 + K*3]      ... + K keypoints (x, y, conf)
  segment   output0 [N, 300, 6 + 32]       ... + 32 mask coefficients
            output1 [N, 32, S/4, S/4]      mask prototypes
  classify  output0 [N, classes]           probabilities
  sequence  probs   [N, classes]           probabilities; input FP32 ``sequences``
            (a non-YOLO network from sequence_models.py, chosen by ``builder``)

    python export/export_onnx.py [--only name ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml

ROOT = Path(__file__).resolve().parents[1]
OPSET = 17
TASKS = ("detect", "classify", "pose", "segment", "sequence")


def tensor_info(vi) -> dict:
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or -1))
    return {
        "name": vi.name,
        "dtype": onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type),
        "shape": dims,
    }


def concrete_outputs(path: Path, inp: dict, item_shape: list) -> list:
    """Run the graph once on a batch of 2 zeros and record each output's real shape."""
    dtype = np.uint8 if inp["dtype"] == "UINT8" else np.float32
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {inp["name"]: np.zeros([2, *item_shape], dtype=dtype)}
    outs = sess.run(None, feed)
    return [{"name": o.name, "dtype": {np.dtype(np.float32): "FLOAT", np.dtype(np.float16): "FLOAT16",
                                       np.dtype(np.int32): "INT32", np.dtype(np.int64): "INT64"}[v.dtype],
             "shape": ["batch", *v.shape[1:]]} for o, v in zip(sess.get_outputs(), outs)]


def inspect_yolo(model) -> dict:
    head = model.model.model[-1]
    names = model.names
    info = {
        "task": model.task,
        "architecture": type(model.model).__name__,
        "head": type(head).__name__,
        "end2end_native": bool(getattr(head, "end2end", False)),
        "num_classes": len(names),
        "names": {int(k): v for k, v in names.items()},
    }
    if model.task == "pose":
        info["kpt_shape"] = [int(v) for v in head.kpt_shape]
    if model.task == "segment":
        info["num_masks"] = int(head.nm)
    return info


def export_yolo(spec: dict, out_dir: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "weights" / spec["weights"]))
    info = inspect_yolo(model)
    if info["task"] != spec["task"]:
        raise SystemExit(f"{spec['name']}: models.yaml says {spec['task']}, weights are {info['task']}")

    kwargs = dict(format="onnx", imgsz=spec["imgsz"], dynamic=True, opset=OPSET, simplify=True,
                  batch=spec["max_batch"], device="cpu")
    # Box tasks: always ship an end-to-end output [N, max_det, 6 + extras], NMS in the graph
    # unless the head is NMS-free in this Ultralytics version.
    embedded_nms = False
    if info["task"] in ("detect", "pose", "segment") and not info["end2end_native"]:
        kwargs["nms"] = True
        embedded_nms = True

    produced = Path(model.export(**kwargs))
    target = out_dir / "model.fp32.onnx"
    produced.replace(target)
    s = spec["imgsz"]
    info.update(embedded_nms=embedded_nms, end2end=info["task"] in ("detect", "pose", "segment"),
                # What the served engine takes after wrap_uint8.py.
                input={"name": "images", "dtype": "UINT8", "dims": [s, s, 3]})
    return info


def export_sequence(spec: dict, out_dir: Path) -> dict:
    import torch

    from sequence_models import BUILDERS, ExportWrapper

    loader, item_shape, names, arch = BUILDERS[spec["builder"]]
    wrapper = ExportWrapper(loader(str(ROOT / "weights" / spec["weights"]))).eval()
    target = out_dir / "model.fp32.onnx"
    # The fused attention kernel (eval fast path) has no ONNX form; export the plain ops.
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.no_grad():
        # Traced at batch 1 so the GRU's initial state is not baked in at a fixed batch size.
        torch.onnx.export(wrapper, (torch.randn(1, *item_shape),), str(target), input_names=["sequences"],
                          output_names=["probs"], dynamic_axes={"sequences": {0: "batch"}, "probs": {0: "batch"}},
                          opset_version=OPSET, dynamo=False)
        dummy = torch.randn(3, *item_shape)
        ref = wrapper(dummy).numpy()
    got = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"]).run(None, {"sequences": dummy.numpy()})[0]
    if not np.allclose(ref, got, atol=1e-4):
        raise SystemExit(f"{spec['name']}: ONNX differs from PyTorch by {np.abs(ref - got).max():.2e}")
    return {"task": "sequence", "architecture": arch, "head": arch, "end2end_native": False,
            "num_classes": len(names), "names": dict(names), "embedded_nms": False, "end2end": False,
            "input": {"name": "sequences", "dtype": "FLOAT", "dims": list(item_shape)}}


def export_one(spec: dict, out_root: Path) -> dict:
    if spec["task"] not in TASKS:
        raise SystemExit(f"{spec['name']}: unknown task {spec['task']}")
    out_dir = out_root / spec["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    info = export_sequence(spec, out_dir) if spec["task"] == "sequence" else export_yolo(spec, out_dir)
    if spec.get("class_names"):
        info["names"].update({int(k): str(v) for k, v in spec["class_names"].items()})

    target = out_dir / "model.fp32.onnx"
    graph = onnx.load(str(target)).graph
    # Shapes of the served graph: the FP32 graph's outputs are the same after wrap_uint8.py.
    fp32_input = tensor_info(graph.input[0])
    item = info["input"]["dims"] if spec["task"] == "sequence" else [3, spec["imgsz"], spec["imgsz"]]
    outputs = concrete_outputs(target, {"name": fp32_input["name"], "dtype": fp32_input["dtype"]}, item)
    meta = {
        **{k: v for k, v in spec.items() if k != "class_names"},
        **info,
        "opset": OPSET,
        "onnx_inputs": [tensor_info(i) for i in graph.input],
        "onnx_outputs": outputs,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--out", default=str(ROOT / "build"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    specs = yaml.safe_load((ROOT / "models.yaml").read_text())["models"]
    for spec in specs:
        if args.only and spec["name"] not in args.only:
            continue
        meta = export_one(spec, Path(args.out))
        print(json.dumps({k: meta.get(k) for k in ("name", "task", "architecture", "head", "embedded_nms",
                                                   "num_classes", "input", "onnx_outputs")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
