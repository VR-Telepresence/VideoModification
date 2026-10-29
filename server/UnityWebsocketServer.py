import asyncio
import logging
import threading

import cv2
import json
import websockets
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

logger = logging.getLogger(__name__)

# FrameProvidingServer
class UnityWebsocketServer:
    frame = None
    __HOST = "0.0.0.0"
    __PORT = 8766

    def __init__(self):
        # Using threading to run the server, that is getting the position data, parallely
        self._ready = threading.Event()

        self._thread = threading.Thread(
            target=self._run_server,
            name="PoseServer",
            daemon=True,
        )
        self._thread.start()

        # Wait until server is listening or startup fails.
        if not self._ready.wait(timeout=10):
            raise TimeoutError("UnitWesocketServer startup timed out")

    # ------- threading --------

    def _run_server(self):
        try:
            asyncio.run(self.__start_server())
        except Exception:
            logger.exception("UnitWesocketServer stopped unexpectedly")
        finally:
            # Unblock constructor even if startup failed.
            self._ready.set()

    def update_frame(self, frame):
        self.frame = frame


    # ------- code for the actual server --------

    async def __start_server(self):
        print(f"[UnityWebsocketServer] Starte WebSocket-Server auf ws://{self.__HOST}:{self.__PORT}...")
        async with websockets.serve(self.__handler, self.__HOST, self.__PORT):
            self._ready.set()
            await asyncio.Future()  # Läuft dauerhaft


    async def __handler(self, websocket):
        print("Unity connected!")
        try:
            while True:
                success, buffer = cv2.imencode(".png", self.frame)
                png_bytes = buffer.tobytes()
                await websocket.send(png_bytes)
                await asyncio.sleep(0.033) # ~30 FPS

        except websockets.exceptions.ConnectionClosed:
            print("Unity disconnected!")