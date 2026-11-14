import threading
from dataclasses import dataclass
from queue import Queue
from typing import Any

import cv2

from video_parallax.server.segmentation_server import SegmentationServer
from video_parallax.time_measuring import TotalMeasurement


@dataclass(frozen=True, slots=True)
class PostprocessJob:
    frame:Any
    frame_nr: int


class PostprocessWorker:
    def __init__(self, detector, output_width, output_height):
        self.__res_imgs:Queue[PostprocessJob] = Queue(1)

        self.__segmentation_server = (
            SegmentationServer(detector=detector) if detector is not None else None
        )
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

        if self.__segmentation_server is not None:
            self.__segmentation_server.update_frame(parallax_frame)

        TotalMeasurement.frame_is_send(frame_nr)
        #self.__writer.write(parallax_frame)