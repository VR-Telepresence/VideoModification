import asyncio
import json
import logging
import threading
from dataclasses import dataclass

import websockets

from server.base_websocket_server import BaseWebsocketServer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# NOTE: Use snapshot()/PoseSnapshot instead of Pos3DData and PosRotation as separate calls can cause frame desync across threads
# Kept it to avoid other scripts not working for now, but both should be removed in the future
@dataclass(frozen=True, slots=True)
class Pos3DData:
    x: float
    y: float
    z: float

@dataclass(frozen=True, slots=True)
class PosRotation:
    yaw: float
    pitch: float

@dataclass(frozen=True, slots=True)
class PoseSnapshot:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    timestamp: float


class PoseServer(BaseWebsocketServer):
    __HOST = "0.0.0.0"
    __PORT = 8765

    def __init__(self):
        self._lock = threading.Lock()
        self.__x = 0.0
        self.__y = 0.0
        self.__z = 0.0
        self.__yaw = 0.0
        self.__pitch = 0.0
        self.__t = 0.0

        _logger = logging.getLogger(self.__class__.__name__)

        super().__init__(
            host="0.0.0.0",
            port="8765",
            name="PoseServer"
        )

    def snapshot(self) -> PoseSnapshot:
        with self._lock:
            return PoseSnapshot(
                self.__x, self.__y, self.__z,
                self.__yaw, self.__pitch, self.__t
            )

    def position_3d(self) -> Pos3DData:
        with self._lock:
            return Pos3DData(self.__x, self.__y, self.__z)

    def camera_rotation(self) -> PosRotation:
        with self._lock:
            return PosRotation(self.__yaw, self.__pitch)

    async def _handler(self, websocket):
        print(f"[PoseServer] Quest 3 connected on {websocket.remote_address}")
        try:
            async for message in websocket:
                data = json.loads(message)
                new_t = data.get("t", 0.0)

                with self._lock:
                    if self.__t < new_t:
                        self.__t = new_t
                        self.__x = data.get("x", self.__x)
                        self.__y = data.get("y", self.__y)
                        self.__z = data.get("z", self.__z)
                        self.__yaw = data.get("yaw", self.__yaw)
                        self.__pitch = data.get("pitch", self.__pitch)
                    else:
                        print("[PoseServer] Old position data discarded")

                logger.debug(f"\r[Pose] Pos: ({self.__x:+.2f}, {self.__y:+.2f}, {self.__z:+.2f})m | Yaw: {self.__yaw:6.1f}° | Pitch: {self.__pitch:6.1f}° | T: {self.__t:.0f}ms", end="")
        except websockets.exceptions.ConnectionClosed:
            print("\n[PoseServer] Quest 3 disconnected.")
        except Exception as e:
            print(f"\n[PoseServer] Error: {e}")