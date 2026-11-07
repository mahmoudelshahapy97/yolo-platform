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
from typing import Dict, List, Optional

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.utils.shared_memory as shm
from tritonclient.utils import InferenceServerException

logger = logging.getLogger(__name__)

ACTIVE_KEY = "models:active"


class ShmSlot:
    def __init__(self, client: grpcclient.InferenceServerClient, max_batch: int, size: int,
                 out_shape=(300, 57), out_dtype=np.float32) -> None:
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

    def infer(self, model: str, version: int, n: int, timeout_s: float = 5.0, out_cols: Optional[int] = None) -> np.ndarray:
        """Run ``n`` frames of the input region; ``out_cols`` is the model's row width (6 detect, 57 pose).

        The output region is sized for the widest model; a narrower model's rows are
        packed at the start of it, so the returned view is reshaped to [n, 300, out_cols].
        """
        rows, cols = self.out_shape[0], out_cols or self.out_shape[1]
        if cols > self.out_shape[1]:
            raise ValueError(f"model output width {cols} exceeds the slot's {self.out_shape[1]}")
        nbytes = n * rows * cols * self.out_dtype.itemsize
        inp = grpcclient.InferInput("images", [n, self.size, self.size, 3], "UINT8")
        inp.set_shared_memory(self.in_name, n * self.size * self.size * 3)
        out = grpcclient.InferRequestedOutput("output0")
        out.set_shared_memory(self.out_name, nbytes)
        self.client.infer(model, [inp], model_version=str(version), outputs=[out], client_timeout=timeout_s)
        flat = self.outputs.reshape(-1)[: n * rows * cols]
        return flat.reshape(n, rows, cols)

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
                timeout_s: float = 5.0, input_name: str = "images", output: str = "output0") -> np.ndarray:
    return infer_outputs(client, model, version, batch, [output], timeout_s, input_name)[0]


def infer_outputs(client: grpcclient.InferenceServerClient, model: str, version: int, batch: np.ndarray,
                  outputs: List[str], timeout_s: float = 5.0, input_name: str = "images") -> List[np.ndarray]:
    """Several outputs in one request (segmentation: detections + mask prototypes), in ``outputs`` order."""
    inp = grpcclient.InferInput(input_name, list(batch.shape), "UINT8" if batch.dtype == np.uint8 else "FP32")
    inp.set_data_from_numpy(np.ascontiguousarray(batch))
    res = client.infer(model, [inp], model_version=str(version),
                       outputs=[grpcclient.InferRequestedOutput(o) for o in outputs], client_timeout=timeout_s)
    return [res.as_numpy(o) for o in outputs]


def model_outputs(model: dict) -> List[str]:
    """Output names of an active model (entries published before 2026-11-06 have none: detection)."""
    return list(model.get("outputs") or ["output0"])


def model_input_name(model: dict) -> str:
    return (model.get("input") or {}).get("name", "images")


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
