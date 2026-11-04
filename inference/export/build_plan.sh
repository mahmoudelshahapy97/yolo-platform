#!/usr/bin/env bash
# Build FP16 TensorRT engines from build/<name>/model.onnx with trtexec from the
# SAME Triton image that serves them (TensorRT plans only load on the TensorRT
# version and GPU architecture that built them).
#
#   inference/export/build_plan.sh [version] [model ...]
#
# Writes model_repository/<name>/<version>/model.plan. Run from the repo root.
# Needs build/<name>/meta.json from export_onnx.py.
set -euo pipefail

VERSION="${1:-1}"; shift || true
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.10-py3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  mapfile -t MODELS < <(ls "$ROOT/build")
fi

for name in "${MODELS[@]}"; do
  meta="$ROOT/build/$name/meta.json"
  imgsz=$(python3 -c "import json;print(json.load(open('$meta'))['imgsz'])")
  maxb=$(python3 -c "import json;print(json.load(open('$meta'))['max_batch'])")
  optb=$(python3 -c "import json;print(json.load(open('$meta'))['opt_batch'])")
  out="$ROOT/model_repository/$name/$VERSION"
  mkdir -p "$out"
  echo "== $name v$VERSION: ${imgsz}px, batch 1/$optb/$maxb, FP16"
  docker run --rm --gpus all -u "$(id -u):$(id -g)" -v "$ROOT:/work" -w /work "$TRITON_IMAGE" \
    /usr/src/tensorrt/bin/trtexec \
      --onnx="build/$name/model.onnx" \
      --saveEngine="model_repository/$name/$VERSION/model.plan" \
      --fp16 \
      --minShapes="images:1x${imgsz}x${imgsz}x3" \
      --optShapes="images:${optb}x${imgsz}x${imgsz}x3" \
      --maxShapes="images:${maxb}x${imgsz}x${imgsz}x3" \
      --memPoolSize=workspace:2048M \
      --skipInference \
      > "$ROOT/build/$name/trtexec-v$VERSION.log" 2>&1 \
    || { echo "trtexec failed for $name, see build/$name/trtexec-v$VERSION.log"; tail -20 "$ROOT/build/$name/trtexec-v$VERSION.log"; exit 1; }
  sha256sum "$out/model.plan" | cut -d' ' -f1 > "$out/model.plan.sha256"
  ls -la "$out/model.plan"
done
