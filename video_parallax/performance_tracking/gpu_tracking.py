import threading
import time
from abc import ABC
from collections import deque
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CudaFrameTracker:
    start: torch.cuda.Event
    end: torch.cuda.Event
    created_at: float


class CudaJobTracking(ABC):

    __cuda_pending: dict[tuple[str, int], CudaFrameTracker] = {}
    __cuda_lock = threading.Lock()
    __cuda_history: dict[str, deque] = {}

    cuda_avg_ms: dict[str, float] = {}
    cuda_avg_ms.update(
        {
            "preproc": float("inf"),
            "parallax":  float("inf"),
        })



    @classmethod
    def add_cuda_measurement(cls, stage, frame_nr, start, end):
        now = time.perf_counter()

        with cls.__cuda_lock:
            cls.__cuda_pending[(stage, frame_nr)] = CudaFrameTracker(
                start, end, now
            )

            cls.__remove_old_cuda_measurements(now)

    @classmethod
    def add_cuda_measurement_preproc(cls, frame_nr, start, end):
        cls.add_cuda_measurement("preproc", frame_nr, start, end)

    @classmethod
    def add_cuda_measurement_parallax(cls, frame_nr, start, end):
        cls.add_cuda_measurement("parallax", frame_nr, start, end)

    @classmethod
    def avg_cuda_measurement_preproc(cls):
        return cls.cuda_avg_ms["preproc"]

    @classmethod
    def avg_cuda_measurement_parallax(cls):
        return cls.cuda_avg_ms["parallax"]



    @classmethod
    def __remove_old_cuda_measurements(cls, now):
        # Called with __cuda_lock held.
        while cls.__cuda_pending:
            oldest_key = next(iter(cls.__cuda_pending))
            oldest = cls.__cuda_pending[oldest_key]

            if (
                now - oldest.created_at <= 2.0
                and len(cls.__cuda_pending) <= 500
            ):
                break
            cls.__cuda_pending.pop(oldest_key)

    @classmethod
    def collect_cuda_measurements(cls):
        # Called periodically by your monitor thread.
        with cls.__cuda_lock:
            pending = list(cls.__cuda_pending.items())

        completed = []

        for key, measurement in pending:
            if measurement.end.query():
                duration = measurement.start.elapsed_time(
                    measurement.end
                )
                completed.append((key, measurement, duration))

        with cls.__cuda_lock:
            for key, measurement, duration in completed:
                # Do not remove a newer measurement with the same key.
                if cls.__cuda_pending.get(key) is not measurement:
                    continue

                cls.__cuda_pending.pop(key)

                stage = key[0]

                if stage not in cls.__cuda_history:
                    cls.__cuda_history[stage] = deque(maxlen=10)

                history = cls.__cuda_history[stage]
                history.append(duration)

                cls.cuda_avg_ms[stage] = sum(history) / len(history)
