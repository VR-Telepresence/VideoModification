import asyncio
import logging
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
                _, jpeg_bytes = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                await websocket.send(jpeg_bytes.tobytes())
        except websockets.exceptions.ConnectionClosed:
            print("Unity disconnected!")

    def update_frame(self, frame):
        self.frame = frame


