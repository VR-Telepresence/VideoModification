from dataclasses import dataclass
from queue import Queue
from typing import Any, Callable
import cv2
from typing_extensions import override
from video_parallax.server.viewport_providing_server import ViewportProvidingServer
from video_parallax.performance_tracking.time_measuring import TotalMeasurement
from video_parallax.worker.base_worker import BaseWorker, Job
from video_parallax.server.segmentation_server import SegmentationServer


@dataclass(frozen=True, slots=True)
class PostprocessJob(Job):
    frame:Any
    frame_nr: int

@dataclass(frozen=True, slots=True)
class PostprocessResult(Job):
    frame:Any
    frame_nr: int




class PostprocessWorker(BaseWorker[PostprocessJob]):
    def __init__(self, detector, output_width, output_height, handle_result_fn:Callable[[PostprocessResult],None]):
        super().__init__(self.__class__.__name__, Queue(1))
        self.__detector = detector

        self.__segmentation_server = (
            SegmentationServer(detector=detector) if detector is not None else None
        )

        self.__writer = self.__create_writer(output_width, output_height)
        self.display_frame = None
        self.counter = 0
        self.__handle_result_fn = handle_result_fn

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

        if self.__segmentation_server is not None:
            self.__segmentation_server.update_frame(parallax_frame)

        TotalMeasurement.frame_is_postproccessed(frame_nr)
        self.__handle_result_fn(PostprocessResult(parallax_frame, frame_nr))
        #self.__writer.write(parallax_frame)

        TotalMeasurement.frame_postprocessing.end_of_loop()

