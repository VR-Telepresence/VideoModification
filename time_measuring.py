import os
import queue
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import time

class LatencyAverager:
    def __init__(self):
        self.__sum = 0.0
        self.__COUNT = 10.0
        self.__last_results = queue.Queue()

    def add_number(self, n):
        if self.__last_results.qsize() == self.__COUNT:
            self.__sum -= self.__last_results.get()
        self.__last_results.put(n)
        self.__sum += n

    def average(self):
        return self.__sum / self.__COUNT




class FPSMeasuring:
    def __init__(self):
        self.__fps = 0.0
        self.__last_result = datetime.now()

        self.__last_result_timestamps = queue.Queue()# only required for avg fps
        self.__oldest_relevant_result_timestamp = datetime.now()# only required for avg fps


    def result_created(self):
        now = datetime.now()
        self.__fps = 1.0 / (now - self.__last_result).total_seconds()
        self.__last_result=now

        self.__last_result_timestamps.put(now)

    def avg_fps(self):
        # make sure only the results of the last second are listed
        while (datetime.now() - self.__oldest_relevant_result_timestamp) > timedelta(seconds = 1):
            self.__oldest_relevant_result_timestamp = self.__last_result_timestamps.get()
        # with only frames from last second
        avg_fps = self.__last_result_timestamps.qsize()
        return avg_fps

    def print_fps(self):
        #print(f"fps: {self.get_fps()}")
        pass

    def get_fps(self):
        return self.__fps




class LatencyMeasuring:
    def __init__(self):
        self.__start_time = None

    def start(self):
        self.__start_time = datetime.now()

    def timediff_seconds(self):
        now = datetime.now()
        return (now - self.__start_time).total_seconds()


    def print_timediff(self):
        #print(f"latency: {self.timediff_seconds()}")
        pass

class SingleProcessMeasurements:
    def __init__(self):

        self.avg_fps = 0
        self.fps = 0
        self.avg_latency = float('inf')
        self.latency = float('inf')

        self.__latency_measurer = LatencyMeasuring()
        self.__fps_measurer = FPSMeasuring()
        self.__latency_averager = LatencyAverager()

    def start_of_loop(self):
        self.__latency_measurer.start()

    def end_of_loop(self):
        self.latency = self.__latency_measurer.timediff_seconds()
        self.__latency_averager.add_number(self.latency)
        self.avg_latency = self.__latency_averager.average()

        self.__fps_measurer.result_created()
        self.fps = self.__fps_measurer.get_fps()
        self.avg_fps = self.__fps_measurer.avg_fps()



class TotalMeasurement:
    _summary_printed = False

    frame_reading = SingleProcessMeasurements()
    frame_parallax_creation = SingleProcessMeasurements()
    frame_postprocessing = SingleProcessMeasurements()
    frame_sending = SingleProcessMeasurements()

    @classmethod
    def print_summary(cls):
        reading = cls.frame_reading
        parallax = cls.frame_parallax_creation
        postproc = cls.frame_postprocessing
        sending = cls.frame_sending

        total_latency = (
                reading.avg_latency
                + parallax.avg_latency
                + postproc.avg_latency
                + sending.avg_latency
        )
        # Explanation for the table below:
        # :	Starts the formatting specification
        # >	Right-align the value
        # 10	Use a minimum width of 10 characters
        # .2f	Display as a float with exactly 2 decimal places

        output = f"""\
        ╔════════════════════════════════════════════╗
        ║             PERFORMANCE MONITOR            ║
        ╠══════════════╦════════════╦════════════════╣
        ║ Stage        ║ FPS        ║ Latency (ms)   ║
        ╠══════════════╬════════════╬════════════════╣
        ║ Reading      ║ {reading.avg_fps:>10.2f} ║ {reading.avg_latency*1000:>14.2f} ║
        ║ Parallax     ║ {parallax.avg_fps:>10.2f} ║ {parallax.avg_latency*1000:>14.2f} ║
        ║ Postproc     ║ {postproc.avg_fps:>10.2f} ║ {postproc.avg_latency*1000:>14.2f} ║
        ║ Sending      ║ {sending.avg_fps:>10.2f} ║ {sending.avg_latency*1000:>14.2f} ║
        ╠══════════════╩════════════╬════════════════╣
        ║ TOTAL LATENCY             ║ {total_latency:>14.2f} ║
        ╚═══════════════════════════╩════════════════╝
        """

        path = Path("./temp/performance.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=path.parent,
                    suffix=".tmp",
                    delete=False,
            ) as f:
                tmp_path = Path(f.name)
                f.write(output)

            for attempt in range(10):
                try:
                    os.replace(tmp_path, path)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.05)

        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)


