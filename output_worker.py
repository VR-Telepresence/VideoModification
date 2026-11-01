import threading
import time
from queue import Queue

import cv2
import torch

from server.viewport_providing_server import ViewportProvidingServer
from time_measuring import FPSMeasuring, LatencyMeasuring


class OutputWorker:
    def __init__(self, detector, output_width, output_height):
        self.__res_imgs = Queue()
        self.__detector = detector
        self.__unityServer = ViewportProvidingServer()
        self.__writer = self.__create_writer(output_width, output_height)
        self.fps_measuring = FPSMeasuring()
        self.latency_measuring = LatencyMeasuring()
        self.display_frame = None

        self._thread = threading.Thread(
            target=self._run_worker,
            name="output_worker",
            daemon=True,
        )
        self._thread.start()

    def add_result_image(self, predicted_img):
        self.__res_imgs.put(predicted_img)

    def __create_writer(self, width, height):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30
        return  cv2.VideoWriter(
            "viewport.mp4",
            fourcc,
            fps,
            (width, height)
        )

    def _run_worker(self):
        while True:
            print("output")

            predicted_img = self.__res_imgs.get()
            self.__handle_parallax_img(predicted_img)


    def __handle_parallax_img(self, predicted_img):
        print(self.__res_imgs.qsize())
        parallax_frame = predicted_img

        self.display_frame = parallax_frame


        if self.__detector is not None:
            self.__detector.submit(parallax_frame)
            parallax_frame = self.__detector.draw(parallax_frame)

        self.fps_measuring.frame_created()
        self.fps_measuring.print_fps()

        self.__unityServer.update_frame(parallax_frame)
        #self.__writer.write(parallax_frame)

