from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from .detector_base import BaseDetector, Detection


class YoloDetector(BaseDetector):
    def __init__(self, weights="yolo26n.pt", only=("person",), conf=0.25, imgsz=256, **kw):
        self.conf, self.imgsz = conf, imgsz
        self.use_cuda = torch.cuda.is_available()
        self.device = 0 if self.use_cuda else "cpu"

        w = Path(weights)
        engine = w if w.suffix == ".engine" else w.with_name(f"{w.stem}_{imgsz}.engine")

        if self.use_cuda and engine.exists():
            self.backend, path = "tensorrt", engine
        else:
            if self.use_cuda:
                print(f"[YoloDetector] No engine at {engine}, falling back to PyTorch. "
                      f"Run export_yolo_trt.py for TensorRT.")
            self.backend, path = "pytorch", w

        self._half = self.backend == "pytorch" and self.use_cuda
        self.model = YOLO(str(path), task="detect")

        name_to_id = {v: k for k, v in self.model.names.items()}
        missing = [n for n in only if n not in name_to_id]
        if missing:
            raise ValueError(f"Unknown classes {missing}. Available: {sorted(name_to_id)}")
        self.class_ids = [name_to_id[n] for n in only]

        self.last_speed = {}
        dummy = np.zeros((imgsz, imgsz, 3), np.uint8)
        for _ in range(3):
            self._predict(dummy)
        print(f"[YoloDetector] backend={self.backend} device={self.device} imgsz={imgsz} model={path.name}")

    def _predict(self, frame):
        return self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, classes=self.class_ids,
            device=self.device, half=self._half, verbose=False,
        )[0]

    def infer(self, frame) -> list[Detection]:
        r = self._predict(frame)
        self.last_speed = r.speed
        if len(r.boxes) == 0:
            return []
        d = r.boxes.data.cpu().numpy()
        names = self.model.names
        return [
            Detection(label=names[int(c)], confidence=float(cf), box=(x1, y1, x2, y2))
            for x1, y1, x2, y2, cf, c in d
        ]