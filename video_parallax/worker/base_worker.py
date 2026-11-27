import logging
import sys
import threading
import traceback
from abc import abstractmethod, ABC

from queue import Queue

from video_parallax.logger import get_logger


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
        self._logger = get_logger(self.__class__.__name__)
        self.__stop_scheduled = False


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
        while not self.__stop_scheduled:
            try:
                job = self.__get_next_job()
                self._handle_job(job)
            except Exception as e:
                self._logger.error(e)
                traceback.print_exc()


    def start(self):
        self._thread.start()
        print(f"[Worker {self.name}] started")

    def stop(self):
        self.__stop_scheduled=True


