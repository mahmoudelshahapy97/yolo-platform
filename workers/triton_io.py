"""Triton I/O for workers: system shared-memory batch slots plus plain gRPC calls.

A ``ShmSlot`` is one in-flight request's input and output buffers, living in
POSIX shared memory that the worker and Triton both map (compose gives the
worker ``ipc: service:triton``). Frames are written into the slot once, by the
decoder's copy, and Triton reads them in place: no gRPC serialisation of
megabytes of pixels per batch.
"""

import json
import logging
import os
import threading
import time
import uuid
from typing import Dict, Optional

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.utils.shared_memory as shm
from tritonclient.utils import InferenceServerException

logger = logging.getLogger(__name__)

ACTIVE_KEY = "models:active"


class ShmSlot:
    def __init__(self, client: grpcclient.InferenceServerClient, max_batch: int, size: int,
                 out_shape=(300, 6), out_dtype=np.float32) -> None:
        self.client = client
        self.max_batch = max_batch
        self.size = size
        tag = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.in_name, self.out_name = f"in_{tag}", f"out_{tag}"
        self.in_bytes = max_batch * size * size * 3
        self.out_shape = out_shape
        self.out_dtype = np.dtype(out_dtype)
        self.out_bytes = max_batch * int(np.prod(out_shape)) * self.out_dtype.itemsize
        self._in = shm.create_shared_memory_region(self.in_name, f"/{self.in_name}", self.in_bytes)
        self._out = shm.create_shared_memory_region(self.out_name, f"/{self.out_name}", self.out_bytes)
        client.register_system_shared_memory(self.in_name, f"/{self.in_name}", self.in_bytes)
        client.register_system_shared_memory(self.out_name, f"/{self.out_name}", self.out_bytes)
        # Writable numpy views straight onto the shared pages.
        self.inputs = shm.get_contents_as_numpy(self._in, np.uint8, [max_batch, size, size, 3])
        self.outputs = shm.get_contents_as_numpy(self._out, self.out_dtype, [max_batch, *out_shape])

    def infer(self, model: str, version: int, n: int, timeout_s: float = 5.0) -> np.ndarray:
        inp = grpcclient.InferInput("images", [n, self.size, self.size, 3], "UINT8")
        inp.set_shared_memory(self.in_name, n * self.size * self.size * 3)
        out = grpcclient.InferRequestedOutput("output0")
        out.set_shared_memory(self.out_name, n * int(np.prod(self.out_shape)) * self.out_dtype.itemsize)
        self.client.infer(model, [inp], model_version=str(version), outputs=[out], client_timeout=timeout_s)
        return self.outputs[:n]

    def close(self) -> None:
        for name, region in ((self.in_name, self._in), (self.out_name, self._out)):
            try:
                self.client.unregister_system_shared_memory(name)
            except Exception:
                pass
            try:
                shm.destroy_shared_memory_region(region)
            except Exception:
                pass


def infer_plain(client: grpcclient.InferenceServerClient, model: str, version: int, batch: np.ndarray,
                timeout_s: float = 5.0) -> np.ndarray:
    inp = grpcclient.InferInput("images", list(batch.shape), "UINT8")
    inp.set_data_from_numpy(np.ascontiguousarray(batch))
    res = client.infer(model, [inp], model_version=str(version), outputs=[grpcclient.InferRequestedOutput("output0")],
                       client_timeout=timeout_s)
    return res.as_numpy("output0")


class ActiveModels:
    """The registry's active versions (Redis hash ``models:active``), refreshed every few seconds."""

    def __init__(self, redis, refresh_s: float = 3.0) -> None:
        self.redis = redis
        self.refresh_s = refresh_s
        self._models: Dict[str, dict] = {}
        self._at = 0.0
        self._lock = threading.Lock()

    def get(self, name: str) -> Optional[dict]:
        now = time.monotonic()
        if now - self._at > self.refresh_s:
            with self._lock:
                if now - self._at > self.refresh_s:
                    raw = self.redis.hgetall(ACTIVE_KEY)
                    models = {}
                    for k, v in raw.items():
                        m = json.loads(v)
                        m["names"] = {int(i): n for i, n in m["names"].items()}
                        models[k] = m
                    self._models = models
                    self._at = now
        return self._models.get(name)

    def wait_for(self, name: str, timeout: float = 300.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._at = 0.0
            m = self.get(name)
            if m:
                return m
            time.sleep(2)
        raise TimeoutError(f"model {name} never became active")


def is_overload(exc: Exception) -> bool:
    if not isinstance(exc, InferenceServerException):
        return False
    msg = (exc.message() or "").lower()
    return "queue" in msg or "timeout" in msg or "deadline" in msg or "unavailable" in msg
