import threading
import time
from abc import abstractmethod, ABC
from dataclasses import dataclass
from queue import Queue
from typing import Any

import cv2
import torch

from video_parallax.server.viewport_providing_server import ViewportProvidingServer
from video_parallax.time_measuring import TotalMeasurement

from typing import TypeVar, Generic


class Job:
    pass

class BaseWorker[T:Job](ABC):
    def __init__(self, name, queue: Queue[T]):
        self.name = name
        self.__jobs = queue
        self._thread = threading.Thread(
            target=self.__run_worker,
            name=f"Worker: {name}",
            daemon=True,
        )
        self._thread.start()

    def add_job(self, job: T):
        if self.__jobs.full():
            self.__jobs.get()
            print(f"[{self.name}] queue is full, removed oldest job")
        self.__jobs.put(job)

    def __get_next_job(self) -> T:
        return self.__jobs.get()


    @abstractmethod
    def _handle_job(self, job: T):
        pass

    def __run_worker(self):
        while True:
            job = self.__get_next_job()
            self._handle_job(job)



