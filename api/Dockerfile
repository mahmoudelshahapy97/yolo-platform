# API image: no torch, no CUDA. Talks to Triton over gRPC.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PROMETHEUS_MULTIPROC_DIR=/tmp/prom
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home app

WORKDIR /app
COPY api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY shared /tmp/shared
RUN pip install --no-cache-dir /tmp/shared && rm -rf /tmp/shared
COPY api/app /app/app
COPY api/requirements-dev.txt /app/requirements-dev.txt
COPY database/migrations /app/migrations
COPY api/entrypoint.sh /app/entrypoint.sh

USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
  CMD curl -fsS http://localhost:8000/health/live || exit 1
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers", "--forwarded-allow-ips", "*", "--timeout-graceful-shutdown", "20"]
