import asyncio
import json
import logging
import threading
from dataclasses import dataclass

import websockets



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



class PoseServer:
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
        #Using threading to run the server, that is getting the position data, parallely
        self._ready = threading.Event()

        self._thread = threading.Thread(
            target=self._run_server,
            name="PoseServer",
            daemon=True,
        )
        self._thread.start()

        # Wait until server is listening or startup fails.
        if not self._ready.wait(timeout=10):
            raise TimeoutError("PoseServer startup timed out")


    # ------- threading --------

    def _run_server(self):
        try:
            asyncio.run(self.__start_server())
        except Exception:
            logger.exception("PoseServer stopped unexpectedly")
        finally:
            # Unblock constructor even if startup failed.
            self._ready.set()

    # ------- code for the actual server --------


    async def __start_server(self):
        print(f"[PoseServer] Starte WebSocket-Server auf ws://{self.__HOST}:{self.__PORT}...")
        async with websockets.serve(self.__handle_pose, self.__HOST, self.__PORT):
            self._ready.set()
            await asyncio.Future()  # Läuft dauerhaft


    async def __handle_pose(self, websocket):
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

if __name__ == "__main__":
    PoseServer()