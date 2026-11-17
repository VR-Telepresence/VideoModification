import threading
import numpy as np
import cv2


def mask_to_polys(mask, min_area=50):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).astype(np.int32) for c in cnts if cv2.contourArea(c) >= min_area]


class BaseDetector:
    def __init__(self, every_n=2, draw_masks=True, draw_boxes=False, style="outline"):
        self.every_n, self.draw_masks, self.draw_boxes, self.style = every_n, draw_masks, draw_boxes, style
        self.frame_id = 0
        self.busy = False
        self.lock = threading.Lock()
        self.latest = None

    def infer(self, frame_bgr):
        raise NotImplementedError

    def submit(self, frame_bgr):
        self.frame_id += 1
        if self.busy or self.frame_id % self.every_n:
            return
        self.busy = True
        threading.Thread(target=self._run, args=(frame_bgr.copy(),), daemon=True).start()

    def _run(self, frame):
        try:
            res = self.infer(frame)
            with self.lock:
                self.latest = res
        except Exception as e:
            print("Detector error:", e)
        finally:
            self.busy = False

    def draw(self, frame):
        with self.lock:
            res = self.latest
        if res is None:
            return frame
        if self.draw_masks and res["polys"]:
            if self.style == "fill":
                overlay = frame.copy()
                for poly in res["polys"]:
                    cv2.fillPoly(overlay, [poly], (0, 255, 0))
                frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
            else:
                cv2.polylines(frame, res["polys"], True, (0, 255, 0), 2)
        if self.draw_boxes:
            for (x1, y1, x2, y2), label in zip(res["boxes"], res["labels"]):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame


# pip install ultralytics