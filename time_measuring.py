from datetime import datetime


class FPSMeasuring:
    __fps = 0
    __last_frame = datetime.now()

    def __init__(self):
        pass


    def frame_created(self):
        now = datetime.now()
        self.__fps = 1 / (now - self.__last_frame).total_seconds()
        self.__last_frame=now

    def print_fps(self):
        #print(f"fps: {self.get_fps()}")
        pass

    def get_fps(self):
        return self.__fps



class LatencyMeasuring:
    __start_time = None

    def start(self):
        self.__start_time = datetime.now()

    def timediff_seconds(self):
        now = datetime.now()
        return (now - self.__start_time).total_seconds()


    def print_timediff(self):
        #print(f"latency: {self.timediff_seconds()}")
        pass


