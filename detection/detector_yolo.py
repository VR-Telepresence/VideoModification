import numpy as np
import torch
from ultralytics import YOLO
from .detector_base import BaseDetector, Detection


class YoloDetector(BaseDetector):
    def __init__(self, weights="yolo26n-seg.pt", only=("vase",), conf=0.25, imgsz=256, **kw):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(weights)
        name_to_id = {v: k for k, v in self.model.names.items()}
        self.class_ids = [name_to_id[n] for n in only]
        self.conf, self.imgsz = conf, imgsz

    def infer(self, frame) -> list[Detection]:
        r = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, classes=self.class_ids,
            device=self.device, half=(self.device == "cuda"), verbose=False,
        )[0]

        if r.masks is None:
            return []

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        return [
            Detection(
                label=self.model.names[int(c)],
                confidence=float(cf),
                polygon=p.astype(np.int32),
                box=tuple(b),
            )
            for p, b, cf, c in zip(r.masks.xy, boxes, confs, clss)
            if len(p) >= 3
        ]