"""Model adapters and the image inference pipeline.

Routes hand decoded BGR images to ``InferenceService``; adapters own every
model-specific step (preprocess, decode, coordinates). Routes never see tensors.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from yolo_platform_shared.decode import decode_classification, decode_detections
from yolo_platform_shared.preprocess import classify_crop, letterbox

from .model_registry import ActiveModel, ModelRegistry
from .triton_client import TritonClient


class ModelNotServing(LookupError):
    pass


@dataclass
class Timings:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    def add(self, other: "Timings") -> None:
        self.preprocess_ms += other.preprocess_ms
        self.inference_ms += other.inference_ms
        self.postprocess_ms += other.postprocess_ms

    def to_dict(self, total_ms: float) -> dict:
        return {"preprocess_ms": round(self.preprocess_ms, 2), "inference_ms": round(self.inference_ms, 2),
                "postprocess_ms": round(self.postprocess_ms, 2), "total_ms": round(total_ms, 2)}


@dataclass
class ImageResult:
    width: int
    height: int
    detections: List[dict] = field(default_factory=list)
    classifications: List[dict] = field(default_factory=list)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


class DetectionAdapter:
    def __init__(self, model: ActiveModel) -> None:
        self.model = model

    def preprocess(self, images: Sequence[np.ndarray]):
        batch = np.empty((len(images), self.model.imgsz, self.model.imgsz, 3), dtype=np.uint8)
        metas = [letterbox(img, self.model.imgsz, out=batch[i])[1] for i, img in enumerate(images)]
        return batch, metas

    def decode(self, outputs: np.ndarray, metas, conf: float, classes: Optional[Sequence[int]], max_det: int):
        return [decode_detections(out, m, self.model.names, conf=conf, classes=classes, max_det=max_det)
                for out, m in zip(outputs, metas)]


class ClassificationAdapter:
    def __init__(self, model: ActiveModel) -> None:
        self.model = model

    def preprocess(self, images: Sequence[np.ndarray]) -> np.ndarray:
        batch = np.empty((len(images), self.model.imgsz, self.model.imgsz, 3), dtype=np.uint8)
        for i, img in enumerate(images):
            classify_crop(img, self.model.imgsz, out=batch[i])
        return batch

    def decode(self, outputs: np.ndarray, top_k: int):
        return [decode_classification(p, self.model.names, top_k=top_k) for p in outputs]


class InferenceService:
    def __init__(self, triton: TritonClient, registry: ModelRegistry) -> None:
        self.triton = triton
        self.registry = registry

    async def _model(self, name: str, task: Optional[str] = None) -> ActiveModel:
        model = await self.registry.active(name)
        if model is None:
            raise ModelNotServing(f"model '{name}' has no active version")
        if task and model.task != task:
            raise ModelNotServing(f"model '{name}' is a {model.task} model, not {task}")
        return model

    async def _run(self, model: ActiveModel, batch: np.ndarray) -> np.ndarray:
        """Split into engine-sized chunks; Triton batches concurrent chunks itself."""
        chunks = [batch[i:i + model.max_batch] for i in range(0, len(batch), model.max_batch)]
        outs = await asyncio.gather(*(self.triton.infer(model.serve_name, 1, c) for c in chunks))
        return np.concatenate(outs)

    async def detect(self, images: List[np.ndarray], model_name: str, conf: float = 0.25,
                     classes: Optional[Sequence[int]] = None, max_det: int = 300,
                     classify_model: Optional[str] = None, classify_classes: Optional[Sequence[int]] = None,
                     top_k: int = 1) -> tuple[ActiveModel, Optional[ActiveModel], List[ImageResult], Timings]:
        model = await self._model(model_name, "detect")
        adapter = DetectionAdapter(model)
        t = Timings()
        s = time.perf_counter()
        batch, metas = await asyncio.to_thread(adapter.preprocess, images)
        t.preprocess_ms += _ms(s)
        s = time.perf_counter()
        outputs = await self._run(model, batch)
        t.inference_ms += _ms(s)
        s = time.perf_counter()
        per_image = adapter.decode(outputs, metas, conf, classes, max_det)
        t.postprocess_ms += _ms(s)

        results = [ImageResult(width=m.orig_w, height=m.orig_h, detections=[d.to_dict() for d in dets])
                   for m, dets in zip(metas, per_image)]

        cls_model = None
        if classify_model:
            # Second stage: classify each (selected) detection crop.
            cls_model = await self._model(classify_model, "classify")
            crops, owners = [], []
            for img_i, (img, dets) in enumerate(zip(images, per_image)):
                for det_i, d in enumerate(dets):
                    if classify_classes is not None and d.class_id not in classify_classes:
                        continue
                    x1, y1, x2, y2 = (int(round(v)) for v in (d.x1, d.y1, d.x2, d.y2))
                    if x2 - x1 >= 4 and y2 - y1 >= 4:
                        crops.append(img[y1:y2, x1:x2])
                        owners.append((img_i, det_i))
            if crops:
                cls_adapter = ClassificationAdapter(cls_model)
                s = time.perf_counter()
                cls_batch = await asyncio.to_thread(cls_adapter.preprocess, crops)
                t.preprocess_ms += _ms(s)
                s = time.perf_counter()
                cls_out = await self._run(cls_model, cls_batch)
                t.inference_ms += _ms(s)
                s = time.perf_counter()
                for (img_i, det_i), top in zip(owners, cls_adapter.decode(cls_out, top_k)):
                    results[img_i].detections[det_i]["classification"] = [c.to_dict() for c in top]
                t.postprocess_ms += _ms(s)
        return model, cls_model, results, t

    async def classify(self, images: List[np.ndarray], model_name: str, top_k: int = 5):
        model = await self._model(model_name, "classify")
        adapter = ClassificationAdapter(model)
        t = Timings()
        s = time.perf_counter()
        batch = await asyncio.to_thread(adapter.preprocess, images)
        t.preprocess_ms += _ms(s)
        s = time.perf_counter()
        outputs = await self._run(model, batch)
        t.inference_ms += _ms(s)
        s = time.perf_counter()
        results = [ImageResult(width=img.shape[1], height=img.shape[0], classifications=[c.to_dict() for c in top])
                   for img, top in zip(images, adapter.decode(outputs, top_k))]
        t.postprocess_ms += _ms(s)
        return model, results, t
