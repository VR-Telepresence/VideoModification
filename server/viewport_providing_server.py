import asyncio
import logging
import struct
import threading

import cv2
import json
import websockets
from ultralytics import YOLO

from server.base_websocket_server import BaseWebsocketServer

model = YOLO("yolo26n-seg.pt")


# FrameProvidingServer
class ViewportProvidingServer(BaseWebsocketServer):
    frame = None
    __HOST = "0.0.0.0"
    __PORT = 8766

    def __init__(self):
        _logger = logging.getLogger(self.__class__.__name__)

        super().__init__(
            host="0.0.0.0",
            port="8766",
            name="Viewport-Server"
        )

    async def _handler(self, websocket):
        print("Unity connected!")
        try:
            while True:
                with self._lock:
                    jpeg, fid = self._jpeg, self._frame_id

                if jpeg is None or fid == last_sent:
                    await asyncio.sleep(0.002)
                    continue

                await websocket.send(struct.pack(">I", fid) + jpeg)  # 4 Byte ID + JPEG
                last_sent = fid
        except websockets.exceptions.ConnectionClosed:
            print("Unity disconnected!")

    def update_frame(self, frame):
        self.frame = frame


