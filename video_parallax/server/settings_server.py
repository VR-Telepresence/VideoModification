import asyncio
import json
import logging
import threading
from dataclasses import dataclass

import websockets

from video_parallax.logger import get_logger
from video_parallax.server.base_websocket_server import BaseWebsocketServer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass(frozen=True, slots=True)
class Viewport:
    angle_h:float
    angle_v:float



class SettingsServer(BaseWebsocketServer):
    __HOST = "0.0.0.0"
    __PORT = 8768


    __detector = "yolo"
    __do_parallax = True
    __view_angle_h = 110
    __view_angle_v = 80
    __mode_was_updated = False


    def viewport(self):
        return Viewport(self.__view_angle_h, self.__view_angle_v)

    def __init__(self):
        _logger = get_logger(self.__class__.__name__)

        super().__init__(
            host=self.__HOST,
            port=self.__PORT,
            name="SettingsServer"
        )

    async def _handler(self, websocket):
        print(f"[{self._name}] verbunden mit {websocket.remote_address}")
        try:
            async for message in websocket:
                data = json.loads(message)

                self.__detector = data.get("detector", "yolo")
                self.__do_parallax = data.get("do_parallax", True)
                self.__view_angle_h, self.__view_angle_v = data.get("view_angle_h", 110), data.get("view_angle_v", 80)

                # Pose Ausgabe im Terminal
                logger.debug(f"\r[Settings] Detector: ({self.__detector} | Do parallax: {self.__do_parallax}° | View Angle horizontal: {self.__view_angle_h:6.1f}° | View Angle vertical: {self.__view_angle_v:.0f}ms", end="")
        except websockets.exceptions.ConnectionClosed:
            print(f"\n[{self._name}] Verbindung getrennt.")
        except Exception as e:
            print(f"\n[{self._name}] Fehler: {e}")
