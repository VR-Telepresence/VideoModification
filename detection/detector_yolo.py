import numpy as np
import torch
from ultralytics import YOLO
from detector_base import BaseDetector


class YoloDetector(BaseDetector):
    def __init__(self, weights="yolo26n-seg.pt", only=("vase",), conf=0.25, imgsz=256, **kw):
        super().__init__(**kw)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(weights)
        name_to_id = {v: k for k, v in self.model.names.items()}
        self.class_ids = [name_to_id[n] for n in only]
        self.conf, self.imgsz = conf, imgsz

    def infer(self, frame):
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                               classes=self.class_ids, device=self.device,
                               half=(self.device == "cuda"), verbose=False)[0]
        return {
            "boxes": r.boxes.xyxy.cpu().numpy().astype(int),
            "labels": [self.model.names[int(c)] for c in r.boxes.cls.cpu().numpy()],
            "polys": [p.astype(np.int32) for p in r.masks.xy] if r.masks is not None else [],
        }