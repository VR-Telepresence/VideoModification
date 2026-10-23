import asyncio
import json
import websockets

HOST = "0.0.0.0"
PORT = 8765


class PoseServer:
    x=0.0
    y=0.0
    z=0.0
    yaw=0.0
    pitch=0.0
    t=0.0

    def __init__(self):
        asyncio.run(self.start_server())

    async def start_server(self):
        print(f"[PoseServer] Starte WebSocket-Server auf ws://{HOST}:{PORT}...")
        async with websockets.serve(self.handle_pose, HOST, PORT):
            await asyncio.Future()  # Läuft dauerhaft


    async def handle_pose(self, websocket):
        print(f"[PoseServer] Quest 3 verbunden von {websocket.remote_address}")
        try:
            async for message in websocket:
                data = json.loads(message)
                if self.t <  data.get("t", 0.0):
                    self.x, self.y, self.z = data.get("x", 0.0), data.get("y", 0.0), data.get("z", 0.0)
                    self.yaw, self.pitch = data.get("yaw", 0.0), data.get("pitch", 0.0)
                else:
                    print("[PoseServer] Old position data discarded")
                # Pose Ausgabe im Terminal
                print(f"\r[Pose] Pos: ({self.x:+.2f}, {self.y:+.2f}, {self.z:+.2f})m | Yaw: {self.yaw:6.1f}° | Pitch: {self.pitch:6.1f}° | T: {self.t:.0f}ms", end="")
        except websockets.exceptions.ConnectionClosed:
            print("\n[PoseServer] Quest 3 getrennt.")
        except Exception as e:
            print(f"\n[PoseServer] Fehler: {e}")

if __name__ == "__main__":
    PoseServer()