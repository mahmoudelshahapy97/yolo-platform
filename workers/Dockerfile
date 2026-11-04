# Workers: camera shards (NVDEC + CuPy), video jobs, event processor. No torch.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
    CUPY_CACHE_DIR=/tmp/cupy
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home app

WORKDIR /app
COPY workers/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY shared /tmp/shared
RUN pip install --no-cache-dir /tmp/shared && rm -rf /tmp/shared
COPY workers /app/workers
COPY workers/requirements-dev.txt /app/requirements-dev.txt

USER app
EXPOSE 9100
