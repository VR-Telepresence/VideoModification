import asyncio
import json
import logging
import threading
from dataclasses import dataclass

import websockets

from server.base_websocket_server import BaseWebsocketServer

logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class Pos3DData:
    x:float
    y:float
    z:float

@dataclass(frozen=True, slots=True)
class PosRotation:
    yaw:float
    pitch:float



class PoseServer(BaseWebsocketServer):
    __HOST = "0.0.0.0"
    __PORT = 8765

    __x=0.0
    __y=0.0
    __z=0.0
    __yaw=0.0
    __pitch=0.0
    __t=0.0

    def position_3d(self):
        return Pos3DData(self.__x,self.__y,self.__z)
    def camera_rotation(self):
        return PosRotation(self.__yaw, self.__pitch)

    def __init__(self):
        _logger = logging.getLogger(self.__class__.__name__)

        super().__init__(
            host="0.0.0.0",
            port="8765",
            name="PoseServer"
        )

    async def _handler(self, websocket):
        print(f"[PoseServer] Quest 3 verbunden von {websocket.remote_address}")
        try:
            async for message in websocket:
                data = json.loads(message)
                if self.__t <  data.get("t", 0.0):
                    self.__x, self.__y, self.__z = data.get("x", 0.0), data.get("y", 0.0), data.get("z", 0.0)
                    self.__yaw, self.__pitch = data.get("yaw", 0.0), data.get("pitch", 0.0)
                else:
                    print("[PoseServer] Old position data discarded")
                # Pose Ausgabe im Terminal
                print(f"\r[Pose] Pos: ({self.__x:+.2f}, {self.__y:+.2f}, {self.__z:+.2f})m | Yaw: {self.__yaw:6.1f}° | Pitch: {self.__pitch:6.1f}° | T: {self.__t:.0f}ms", end="")
        except websockets.exceptions.ConnectionClosed:
            print("\n[PoseServer] Quest 3 getrennt.")
        except Exception as e:
            print(f"\n[PoseServer] Fehler: {e}")
