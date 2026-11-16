from dataclasses import dataclass
from queue import Queue
from typing import Any
import cv2
from typing_extensions import override
from video_parallax.server.viewport_providing_server import ViewportProvidingServer
from video_parallax.time_measuring import TotalMeasurement
from video_parallax.worker.base_worker import BaseWorker, Job


@dataclass(frozen=True, slots=True)
class PostprocessJob(Job):
    frame:Any
    frame_nr: int


class PostprocessWorker(BaseWorker[PostprocessJob]):
    def __init__(self, detector, output_width, output_height):
        super().__init__(self.__class__.__name__, Queue(1))
        self.__detector = detector
        self.__unityServer = ViewportProvidingServer()
        self.__writer = self.__create_writer(output_width, output_height)
        self.display_frame = None
        self.counter = 0

    def __create_writer(self, width, height):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30
        return  cv2.VideoWriter(
            "viewport.mp4",
            fourcc,
            fps,
            (width, height)
        )
    @override
    def _handle_job(self, job: PostprocessJob):
        TotalMeasurement.frame_postprocessing.start_of_loop()

        parallax_frame = job.frame
        frame_nr = job.frame_nr

        self.display_frame = parallax_frame


        if self.__detector is not None:
            self.__detector.submit(parallax_frame)
            parallax_frame = self.__detector.draw(parallax_frame)

        self.__unityServer.update_frame(parallax_frame, frame_nr)
        #self.__writer.write(parallax_frame)

        TotalMeasurement.frame_is_postproccessed(frame_nr)
        TotalMeasurement.frame_postprocessing.end_of_loop()

