import asyncio
import json
import logging
import threading

import cv2
import websockets

from detection import create_detector
from server.base_websocket_server import BaseWebsocketServer


class SegmentationServer(BaseWebsocketServer):
    def __init__(
        self,
        detector: str = "yolo",
        weights: str = "yolo26n-seg.pt",
        classes: tuple[str, ...] = ("person",),
        conf: float = 0.25,
        imgsz: int = 256,
        host: str = "0.0.0.0",
        port: str = "8767",
        jpeg_quality: int = 75,
        **detector_kw,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._jpeg_quality = jpeg_quality
        self._detector = (
            None if detector == "none"
            else create_detector(detector, weights=weights, only=classes,
                                 conf=conf, imgsz=imgsz, **detector_kw)
        )
        self._frame = None
        self._frame_id = 0
        self._lock = threading.Lock()

        super().__init__(host=host, port=port, name="Segmentation-Server")

    def update_frame(self, frame):
        with self._lock:
            self._frame = frame.copy()
            self._frame_id += 1

    def _process(self, frame, frame_id):
        h, w = frame.shape[:2]
        detections = self._detector.infer(frame) if self._detector is not None else []
        return json.dumps({
            "type": "segmentation",
            "frame_id": frame_id,
            "objects": [d.to_dict(w, h) for d in detections],
        })

    async def _handler(self, websocket):
        print(f"[{self._name}] Client connected: {websocket.remote_address}")
        last_sent = -1
        try:
            while True:
                with self._lock:
                    frame, fid = self._frame, self._frame_id

                if frame is None or fid == last_sent:
                    await asyncio.sleep(0.002)
                    continue

                payload = await asyncio.to_thread(self._process, frame, fid)
                last_sent = fid
                await websocket.send(payload)
        except websockets.exceptions.ConnectionClosed:
            print(f"[{self._name}] Client disconnected.")
        except Exception:
            self._logger.exception(f"[{self._name}] Handler Error")