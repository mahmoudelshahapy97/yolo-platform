"""Inspect each model in models.yaml and export it to ONNX.

For every model this writes build/<name>/model.onnx plus build/<name>/meta.json.
meta.json records what the inspection found: task, class names, architecture,
whether the head is end-to-end (NMS-free YOLO26, or NMS embedded at export), and
the ONNX input/output tensors. Everything downstream (the Triton config, the API
adapters, the parity check) reads meta.json and never guesses at tensor shapes.

    python export/export_onnx.py [--only name ...]
"""

import argparse
import json
import sys
from pathlib import Path

import onnx
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OPSET = 17


def tensor_info(vi) -> dict:
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or -1))
    return {
        "name": vi.name,
        "dtype": onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type),
        "shape": dims,
    }


def inspect(model: YOLO) -> dict:
    head = model.model.model[-1]
    names = model.names
    return {
        "task": model.task,
        "architecture": type(model.model).__name__,
        "head": type(head).__name__,
        "end2end_native": bool(getattr(head, "end2end", False)),
        "num_classes": len(names),
        "names": {int(k): v for k, v in names.items()},
    }


def export_one(spec: dict, out_root: Path) -> dict:
    weights = ROOT / "weights" / spec["weights"]
    model = YOLO(str(weights))
    info = inspect(model)
    if info["task"] != spec["task"]:
        raise SystemExit(f"{spec['name']}: models.yaml says {spec['task']}, weights are {info['task']}")

    kwargs = dict(format="onnx", imgsz=spec["imgsz"], dynamic=True, opset=OPSET, simplify=True,
                  batch=spec["max_batch"], device="cpu")
    # Detection: always ship an end-to-end output [N, max_det, 6] = x1 y1 x2 y2 conf cls.
    # YOLO26 is NMS-free by design; older heads (yolov8 PPE) get NMS embedded.
    embedded_nms = False
    if info["task"] == "detect" and not info["end2end_native"]:
        kwargs["nms"] = True
        embedded_nms = True

    produced = Path(model.export(**kwargs))
    out_dir = out_root / spec["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "model.fp32.onnx"
    produced.replace(target)

    graph = onnx.load(str(target)).graph
    meta = {
        **spec,
        **info,
        "embedded_nms": embedded_nms,
        "end2end": info["task"] == "detect",
        "opset": OPSET,
        "onnx_inputs": [tensor_info(i) for i in graph.input],
        "onnx_outputs": [tensor_info(o) for o in graph.output],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--out", default=str(ROOT / "build"))
    args = parser.parse_args()
    specs = yaml.safe_load((ROOT / "models.yaml").read_text())["models"]
    for spec in specs:
        if args.only and spec["name"] not in args.only:
            continue
        meta = export_one(spec, Path(args.out))
        print(json.dumps({k: meta[k] for k in ("name", "task", "architecture", "head", "end2end_native",
                                               "embedded_nms", "num_classes", "onnx_inputs", "onnx_outputs")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
