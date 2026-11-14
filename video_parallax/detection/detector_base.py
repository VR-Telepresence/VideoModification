from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np


def mask_to_polys(mask, min_area=50):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).astype(np.int32) for c in cnts if cv2.contourArea(c) >= min_area]


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    polygon: np.ndarray
    box: tuple[int, int, int, int] | None = None

    def to_dict(self, w: int, h: int) -> dict:
        return {
            "label": self.label,
            "confidence": float(self.confidence),
            "polygon": [{"x": float(x) / w, "y": float(y) / h} for x, y in self.polygon],
        }

class BaseDetector(ABC):
    @abstractmethod
    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Caller handles threading"""