from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    def to_dict(self, w: int, h: int) -> dict:
        x1, y1, x2, y2 = self.box
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 3),
            # normalized 0..1, order: x1, y1, x2, y2
            "box": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
        }


class BaseDetector(ABC):
    @abstractmethod
    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Caller handles threading"""