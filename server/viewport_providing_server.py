import asyncio
import logging
import struct
import threading

import cv2
import json
import websockets
from ultralytics import YOLO

from server.base_websocket_server import BaseWebsocketServer
from time_measuring import TotalMeasurement

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

        self._loop = None
        self._frame_updated = None

    async def _handler(self, websocket):
        print("Unity connected!")
        self._loop = asyncio.get_running_loop()
        self._frame_updated = asyncio.Event()
        try:
            while True:

                await self._frame_updated.wait()
                TotalMeasurement.frame_sending.start_of_loop()
                self._frame_updated.clear()

                _, jpeg_bytes = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                await websocket.send(jpeg_bytes.tobytes())
                TotalMeasurement.frame_sending.end_of_loop()

        except websockets.exceptions.ConnectionClosed:
            print("Unity disconnected!")

    def update_frame(self, frame):
        self.frame = frame

        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._frame_updated.set)


