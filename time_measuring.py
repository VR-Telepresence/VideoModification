import queue
from dataclasses import dataclass
from datetime import datetime, timedelta




class LatencyAverager:
    def __init__(self):
        self.__sum = 0
        self.__COUNT = 10
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
        self.__fps = 0
        self.__last_result = datetime.now()

        self.__last_result_timestamps = queue.Queue()# only required for avg fps
        self.__oldest_relevant_result_timestamp = datetime.now()# only required for avg fps


    def result_created(self):
        now = datetime.now()
        self.__fps = 1 / (now - self.__last_result).total_seconds()
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
        print(f"fps: {self.get_fps()}")

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
        print(f"latency: {self.timediff_seconds()}")

class SingleProcessMeasurements:
    def __init__(self):

        self.avg_fps = None
        self.fps = None
        self.avg_latency = None
        self.latency = None

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
    frame_reading = SingleProcessMeasurements()
    frame_parallax_creation = SingleProcessMeasurements()
    frame_postprocessing = SingleProcessMeasurements()
    frame_sending = SingleProcessMeasurements()

    @classmethod
    def print_summary(cls):


        print(f"""  reading:  avg_fps:{round(cls.frame_reading.avg_fps, 4)} latency:{cls.frame_reading.avg_latency}
                    parallax: avg_fps:{round(cls.frame_parallax_creation.avg_fps, 4)} latency:{cls.frame_parallax_creation.avg_latency}
                    postproc: avg_fps:{round(cls.frame_postprocessing.avg_fps, 4)} latency:{cls.frame_postprocessing.avg_latency}
                    sending:  avg_fps:{round(cls.frame_sending.avg_fps, 4)} latency:{cls.frame_sending.avg_latency}
                    total_latency:{cls.frame_reading.avg_latency + cls.frame_parallax_creation.avg_fps + cls.frame_postprocessing.avg_fps + cls.frame_sending.avg_fps}
                """)

