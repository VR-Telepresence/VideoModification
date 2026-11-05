import threading
import time
from queue import Queue

import cv2
import torch

from server.viewport_providing_server import ViewportProvidingServer
from time_measuring import TotalMeasurement


class OutputWorker:
    def __init__(self, unity_server):
        self.__res_imgs = Queue()
        self.__unityServer = unity_server
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
            TotalMeasurement.frame_postprocessing.start_of_loop()
            predicted_img = self.__res_imgs.get()
            self.__handle_parallax_img(predicted_img)
            TotalMeasurement.frame_postprocessing.end_of_loop()

    def __handle_parallax_img(self, predicted_img):
        self.display_frame = predicted_img
        self.__unityServer.update_frame(predicted_img)
        #self.__writer.write(parallax_frame)

