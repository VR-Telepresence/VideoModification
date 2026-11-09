import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Any

import cv2
import torch

from server.viewport_providing_server import ViewportProvidingServer
from time_measuring import TotalMeasurement


@dataclass(frozen=True, slots=True)
class PostprocessJob:
    frame:Any
    frame_nr: int


class PostprocessWorker:
    def __init__(self, detector, output_width, output_height):
        self.__res_imgs:Queue[PostprocessJob] = Queue(1)
        self.__detector = detector
        self.__unityServer = ViewportProvidingServer()
        self.__writer = self.__create_writer(output_width, output_height)
        self.display_frame = None
        self.counter = 0

        self._thread = threading.Thread(
            target=self._run_worker,
            name="output_worker",
            daemon=True,
        )
        self._thread.start()

    def add_result_image(self, predicted_img, frame_nr):
        self.__res_imgs.put(PostprocessJob(predicted_img, frame_nr))

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
            post_process_job = self.__res_imgs.get()
            predicted_img = post_process_job.frame

            TotalMeasurement.frame_postprocessing.start_of_loop()
            self.__handle_parallax_img(predicted_img, post_process_job.frame_nr)
            TotalMeasurement.frame_is_postproccessed(post_process_job.frame_nr)
            TotalMeasurement.frame_postprocessing.end_of_loop()

    def __handle_parallax_img(self, predicted_img, frame_nr):
        parallax_frame = predicted_img

        self.display_frame = parallax_frame


        if self.__detector is not None:
            self.__detector.submit(parallax_frame)
            parallax_frame = self.__detector.draw(parallax_frame)

        self.__unityServer.update_frame(parallax_frame, frame_nr)
        #self.__writer.write(parallax_frame)

