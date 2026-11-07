#!/usr/bin/env bash
# Build TensorRT engines (FP16 unless meta.json says precision: fp32) from build/<name>/model.onnx with trtexec from the
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
  # Input name and per-item dims (images: SxSx3 UINT8; sequences: TxVxC FP32), batch range, precision.
  read -r inp dims maxb optb prec < <(python3 -c "
import json; m = json.load(open('$meta'))
i = m.get('input') or {'name': 'images', 'dims': [m['imgsz'], m['imgsz'], 3]}
print(i['name'], 'x'.join(map(str, i['dims'])), m['max_batch'], m['opt_batch'], m.get('precision', 'fp16'))")
  flags=(); [ "$prec" = fp16 ] && flags+=(--fp16)
  out="$ROOT/model_repository/$name/$VERSION"
  mkdir -p "$out"
  echo "== $name v$VERSION: $inp ${dims}, batch 1/$optb/$maxb, ${prec^^}"
  docker run --rm --gpus all -u "$(id -u):$(id -g)" -v "$ROOT:/work" -w /work "$TRITON_IMAGE" \
    /usr/src/tensorrt/bin/trtexec \
      --onnx="build/$name/model.onnx" \
      --saveEngine="model_repository/$name/$VERSION/model.plan" \
      "${flags[@]}" \
      --minShapes="$inp:1x$dims" \
      --optShapes="$inp:${optb}x$dims" \
      --maxShapes="$inp:${maxb}x$dims" \
      --memPoolSize=workspace:2048M \
      --skipInference \
      > "$ROOT/build/$name/trtexec-v$VERSION.log" 2>&1 \
    || { echo "trtexec failed for $name, see build/$name/trtexec-v$VERSION.log"; tail -20 "$ROOT/build/$name/trtexec-v$VERSION.log"; exit 1; }
  sha256sum "$out/model.plan" | cut -d' ' -f1 > "$out/model.plan.sha256"
  ls -la "$out/model.plan"
done
