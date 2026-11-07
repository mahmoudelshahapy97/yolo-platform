"""Async gRPC client for Triton, shared by every request in an API process."""

import logging
from typing import Dict, List, Optional

import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException

logger = logging.getLogger(__name__)


class TritonUnavailable(Exception):
    """Triton refused or dropped the request (queue full, timeout, not ready)."""


class TritonClient:
    def __init__(self, url: str, timeout_s: float) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self._client: Optional[grpcclient.InferenceServerClient] = None

    @property
    def client(self) -> grpcclient.InferenceServerClient:
        if self._client is None:
            self._client = grpcclient.InferenceServerClient(self.url)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def live(self) -> bool:
        try:
            return await self.client.is_server_live()
        except Exception:
            return False

    async def ready(self) -> bool:
        try:
            return await self.client.is_server_ready()
        except Exception:
            return False

    async def model_ready(self, name: str, version: str = "") -> bool:
        try:
            return await self.client.is_model_ready(name, version)
        except Exception:
            return False

    async def repository_index(self) -> list:
        idx = await self.client.get_model_repository_index(as_json=True)
        return idx.get("models", [])

    async def load(self, name: str) -> None:
        """Load ``name`` from the serving repository (explicit model control)."""
        await self.client.load_model(name)

    async def unload(self, name: str) -> None:
        await self.client.unload_model(name)

    async def infer(self, model: str, version: int, batch: np.ndarray, output: str = "output0",
                    input_name: str = "images") -> np.ndarray:
        """One output of ``model`` on ``batch``."""
        return (await self.infer_many(model, version, batch, [output], input_name))[0]

    async def infer_many(self, model: str, version: int, batch: np.ndarray, outputs: List[str],
                         input_name: str = "images") -> List[np.ndarray]:
        """Several outputs (e.g. segmentation: detections + mask prototypes) in one request, in ``outputs`` order."""
        inp = grpcclient.InferInput(input_name, list(batch.shape), "UINT8" if batch.dtype == np.uint8 else "FP32")
        inp.set_data_from_numpy(np.ascontiguousarray(batch))
        try:
            res = await self.client.infer(
                model, [inp], model_version=str(version),
                outputs=[grpcclient.InferRequestedOutput(o) for o in outputs],
                client_timeout=self.timeout_s,
            )
        except InferenceServerException as exc:
            msg = exc.message() or ""
            if "queue" in msg.lower() or "timeout" in msg.lower() or "unavailable" in msg.lower() or exc.status() in (
                "StatusCode.UNAVAILABLE", "StatusCode.DEADLINE_EXCEEDED", "StatusCode.RESOURCE_EXHAUSTED"
            ):
                raise TritonUnavailable(msg) from exc
            raise
        return [res.as_numpy(o) for o in outputs]

    async def statistics(self, name: str = "") -> Dict:
        return await self.client.get_inference_statistics(name, as_json=True)
