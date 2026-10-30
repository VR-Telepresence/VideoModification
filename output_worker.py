import threading
import time
from queue import Queue

import cv2
import torch

from server.viewport_providing_server import ViewportProvidingServer
from time_measuring import FPSMeasuring, LatencyMeasuring


class OutputWorker:
    def __init__(self, imgs_queue: Queue, detector, writer):
        self.__imgs = imgs_queue
        self.__detector = detector
        self.__unityServer = ViewportProvidingServer()
        self.__writer = writer
        self.fps_measuring = FPSMeasuring()
        self.latency_measuring = LatencyMeasuring()

        self._thread = threading.Thread(
            target=self._run_worker,
            name="output_worker",
            daemon=True,
        )
        self._thread.start()



    def _run_worker(self):
        while True:
            print("output")

            predicted_img = self.__imgs.get()
            self.handle_parallax_img(predicted_img)




    def handle_parallax_img(self, predicted_img):
        print(self.__imgs.qsize())
        parallax_frame = predicted_img



        if self.__detector is not None:
            self.__detector.submit(parallax_frame)
            parallax_frame = self.__detector.draw(parallax_frame)

        self.fps_measuring.frame_created()
        self.fps_measuring.print_fps()

        self.__unityServer.update_frame(parallax_frame)

        # show parallax to user
        cv2.imshow("", parallax_frame)
        self.__writer.write(parallax_frame)

