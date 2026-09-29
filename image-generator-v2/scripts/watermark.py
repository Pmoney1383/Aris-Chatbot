"""
Watermark detector: LAION-5B's pretrained classifier (EfficientNet-B3 with a
2-class head, trained on watermarked vs. clean images). Weights come from
https://github.com/LAION-AI/LAION-5B-WatermarkDetection (models/watermark_model_v1.pt).

Runs on the GPU when available. Worker threads call score() concurrently; a
background thread gathers pending requests into batches so the GPU sees a
few large forward passes instead of many single-image ones.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future

import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

import config

_PREPROCESS = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _build_model() -> nn.Module:
    # Same architecture as LAION's example_use.py. efficientnet_b3 and their
    # 'efficientnet_b3a' share weights layout; pretrained ImageNet weights are
    # skipped because the checkpoint overwrites every parameter anyway.
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1536, out_features=625),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2),
    )
    state_dict = torch.load(config.WATERMARK_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.eval()


class WatermarkDetector:
    MAX_BATCH = 32

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = _build_model().to(self.device)
        self._queue: queue.Queue[tuple[torch.Tensor, Future]] = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def score(self, img: Image.Image) -> float:
        """Probability in [0, 1] that the image carries a watermark."""
        tensor = _PREPROCESS(img.convert("RGB"))
        fut: Future = Future()
        self._queue.put((tensor, fut))
        return fut.result()

    def _loop(self):
        while True:
            items = [self._queue.get()]
            while len(items) < self.MAX_BATCH:
                try:
                    items.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            try:
                batch = torch.stack([t for t, _ in items]).to(self.device)
                with torch.inference_mode():
                    probs = torch.softmax(self.model(batch), dim=1)[:, 0].tolist()
                for (_, fut), p in zip(items, probs):
                    fut.set_result(p)
            except Exception as e:
                for _, fut in items:
                    fut.set_exception(e)
