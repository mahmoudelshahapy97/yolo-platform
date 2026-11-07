# Export toolbox: Ultralytics (CPU torch) + onnx, for .pt -> ONNX and parity checks.
# TensorRT engines are NOT built here; build_plan.sh uses trtexec from the pinned
# Triton image so the engine and the runtime share one TensorRT version.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 torchvision==0.23.0 \
    && pip install --no-cache-dir \
        ultralytics==8.4.178 onnx==1.19.0 onnxslim==0.1.71 onnxruntime==1.23.2 \
        pyyaml==6.0.3 numpy==2.2.6 tritonclient[grpc]==2.61.0

# ByteTrack's assignment solver: parity.py builds pose windows with Ultralytics' tracker.
RUN pip install --no-cache-dir lap==0.5.12

WORKDIR /work
ENV YOLO_CONFIG_DIR=/tmp/ultralytics PYTHONDONTWRITEBYTECODE=1
