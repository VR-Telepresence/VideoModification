import asyncio
import logging
import threading
from abc import ABCMeta, abstractmethod

import websockets



class BaseWebsocketServer(object, metaclass=ABCMeta):


    @abstractmethod
    def __init__(self, host:str, port:str, name:str):
        self._host = host
        self._port = port
        self._name = name
        self.__logger = logging.getLogger(self.__class__.__name__)

        #Using threading to run the server, that is getting the position data, parallely
        self._ready = threading.Event()

        self._thread = threading.Thread(
            target=self._run_server,
            name=name,
            daemon=True,
        )
        self._thread.start()

        # Wait until server is listening or startup fails.
        if not self._ready.wait(timeout=10):
            raise TimeoutError(f"{self._name} startup timed out")


    # ------- threading --------

    def _run_server(self):
        try:
            asyncio.run(self.__start_server())
        except Exception:
            self.__logger.exception(f"{self._name} stopped unexpectedly")
        finally:
            # Unblock constructor even if startup failed.
            self._ready.set()

    # ------- code for the actual server --------


    async def __start_server(self):
        print(f"[{self._name}] Start WebSocket-Server on ws://{self._host}:{self._port}...")
        async with websockets.serve(self._handler, self._host, self._port):
            self._ready.set()
            await asyncio.Future()  # Läuft dauerhaft

    @abstractmethod
    async def _handler(self, websocket):
        pass
