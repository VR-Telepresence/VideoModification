import threading
from abc import abstractmethod, ABC

from queue import Queue

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

    def start(self):
        self._thread.start()

