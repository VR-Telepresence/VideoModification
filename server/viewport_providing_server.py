import asyncio
import logging
import struct
import threading

import cv2
import json
import websockets
from ultralytics import YOLO

from detection import create_detector
from server.base_websocket_server import BaseWebsocketServer
from time_measuring import TotalMeasurement

model = YOLO("yolo26n-seg.pt")


# FrameProvidingServer
class ViewportProvidingServer(BaseWebsocketServer):
    def __init__(
            self,
            host: str = "0.0.0.0",
            port: str = "8766",
            detector: str = "none",
            weights: str = "yolo26n-seg.pt",
            classes: tuple[str, ...] = ("person",),
            conf: float = 0.25,
            imgsz: int = 256,
            jpeg_quality: int = 75,
            **detector_kw,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._jpeg_quality = jpeg_quality
        self._lock = threading.Lock()

        self._jpeg: bytes | None = None
        self._frame_id = 0

        self._seg_json: bytes = b'{"objects":[]}'
        self._seg_frame_id = 0

        self._detector = (
            None if detector == "none"
            else create_detector(detector, weights=weights, only=classes,
                                 conf=conf, imgsz=imgsz, **detector_kw)
        )

        self._pending = None  # (frame, id) fuer den Detection-Worker
        self._new_frame = threading.Event()
        if self._detector is not None:
            threading.Thread(target=self._detection_worker, daemon=True,
                             name="detection").start()

        super().__init__(
            host=host,
            port=port,
            name="Viewport-Server"
        )

    def update_frame(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
            self._frame_id += 1
            fid = self._frame_id
            if self._detector is not None:
                self._pending = (frame.copy(), fid)
        if self._detector is not None:
            self._new_frame.set()

    def _detection_worker(self):
        while True:
            self._new_frame.wait()
            self._new_frame.clear()
            with self._lock:
                job, self._pending = self._pending, None
            if job is None:
                continue
            frame, fid = job
            try:
                h, w = frame.shape[:2]
                detections = self._detector.infer(frame)
                data = json.dumps({"objects": [d.to_dict(w, h) for d in detections]}).encode()
                with self._lock:
                    self._seg_json, self._seg_frame_id = data, fid
            except Exception:
                self._logger.exception("Detection error")

        self._loop = None
        self._frame_updated = None

    async def _handler(self, websocket):
        print(f"[{self._name}] Client connected: {websocket.remote_address}")
        self._loop = asyncio.get_running_loop()
        self._frame_updated = asyncio.Event()
        last_sent = -1
        try:
            while True:
                with self._lock:
                    jpeg, fid = self._jpeg, self._frame_id
                    seg, seg_fid = self._seg_json, self._seg_frame_id

                if jpeg is None or fid == last_sent:
                    await asyncio.sleep(0.002)
                    continue

                TotalMeasurement.frame_sending.start_of_loop()

                header = struct.pack(">III", fid, seg_fid, len(seg))
                await websocket.send(header + seg + jpeg)
                TotalMeasurement.frame_sending.end_of_loop()
                last_sent = fid
        except websockets.exceptions.ConnectionClosed:
            print(f"[{self._name}] Client disconnected.")

        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._frame_updated.set)