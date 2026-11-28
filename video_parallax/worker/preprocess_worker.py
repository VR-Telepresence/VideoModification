import logging
import threading
from dataclasses import dataclass
from queue import Queue
from time import perf_counter, perf_counter_ns
from typing import Any, Callable
import torch
from pytorch360convert import e2p
from typing_extensions import override

import parameters as params
from video_parallax.gpu_manager import GPUManager

from video_parallax.performance_tracking.time_measuring import TotalMeasurement
from video_parallax.worker.base_worker import BaseWorker, Job
from predict_view import getPositionVector
from video_parallax.server.posetracker_server import PoseServer


@dataclass(frozen=True, slots=True)
class PreprocessJob(Job):
    frame:Any
    frame_nr: int

@dataclass(frozen=True, slots=True)
class PreprocessResult:
    img:torch.Tensor
    pos:torch.Tensor
    frame_nr: int
    ready_event: torch.Event
    stream: torch.cuda.Stream | None


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class PreprocessWorker(BaseWorker[PreprocessJob]):
    def __init__(self, pose_dims, width, height, handle_result_fn:Callable[[PreprocessResult],None], gpu_manager: GPUManager|None):
        super().__init__(self.__class__.__name__, Queue(1))
        self.pose_server = PoseServer()
        self.__handle_result_fn = handle_result_fn

        self.width = width
        self.height = height

        self.theta = 0
        self.radius = 0.06
        self.yaw = 0

        self.pose_dims = pose_dims

        # using seperate stream to not ne held up by other threads gpu executions and syncronisation times, improved preprocessers latency by from 30 to 15ms
        self.gpu_manager = gpu_manager
        self._stream = None

        self.events = None
        self._pinned_frame = None

    @override
    def _handle_job(self, job: PreprocessJob):
        if self.gpu_manager is not None:
            self._stream = self.gpu_manager.get_stream()
            self.__extract_viewport_and_pass_result_gpu(job)
        else:
            self.__extract_viewport_and_pass_result(job)


    def __extract_viewport_and_pass_result_gpu(self, job: PreprocessJob):

        stream = self._stream
        with torch.cuda.stream(stream):
            logger.debug(f"""
                {threading.current_thread().name},
                "CUDA stream:",
                {torch.cuda.current_stream().cuda_stream}"""
                         )

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)

            self.__extract_viewport_and_pass_result(job)

            end.record(stream)
            TotalMeasurement.cuda_job_tracking.add_cuda_measurement_preproc(job.frame_nr, start, end)


    def __extract_viewport_and_pass_result(self, job: PreprocessJob):
        frame_nr = job.frame_nr

        TotalMeasurement.frame_preproc.start_of_loop()
        TotalMeasurement.frame_is_read(frame_nr)

        rotation = self.pose_server.camera_rotation()


        viewport_img = self.__extract_viewport(job.frame, rotation.yaw, -rotation.pitch)


        self.__pass_frame_to_parallax_worker(viewport_img, frame_nr)

        TotalMeasurement.frame_is_preproccessed(frame_nr)
        TotalMeasurement.frame_preproc.end_of_loop()


    def __pass_frame_to_parallax_worker(self, img_input, frame_nr):
        # Preprocess Input
        pos3D = self.pose_server.position_3d()
        print(pos3D.x, pos3D.y, pos3D.z)
        max_radius = 0.07
        x = max(min(-pos3D.x, max_radius), -max_radius)
        y = max(min(pos3D.y, max_radius), -max_radius)
        z = max(min(-pos3D.z, max_radius), -max_radius)

        pos = getPositionVector(x, y, z, 0, 0, 0, pose_dims=self.pose_dims).unsqueeze(
            0).to(params.DEVICE, non_blocking=True)

        if self._stream is not None:
            ready_event = torch.cuda.Event()
            ready_event.record(self._stream)
        else:
            ready_event = None

        self.__handle_result_fn(PreprocessResult(img_input, pos, frame_nr, ready_event, self._stream))


    def __extract_viewport(self, frame, yaw, pitch=0, roll=0):
        TotalMeasurement.frame_preproc_viewport_extraction.start_of_loop()
        if self._pinned_frame is None:
            self._pinned_frame = torch.empty(
                frame.shape,
                dtype=torch.uint8,
                pin_memory=True if params.DEVICE == "cuda" else False,
            )

        self._pinned_frame.copy_(torch.from_numpy(frame))

        frame_tensor =  (self._pinned_frame.
            to(params.DEVICE, non_blocking=True).
            permute(2, 0, 1)  # HWC -> CHW
            .float()
            / 255.0
        )
        viewport_tensor = e2p(
            frame_tensor,
            fov_deg=(110, 80), #horizontal, vertical
            h_deg=yaw,  # Yaw: center
            v_deg=pitch,  # Pitch: center
            out_hw=(self.height, self.width),
            mode="bilinear",
        )

        # TODO: was this line does again
        img_input = viewport_tensor[[2, 1, 0], :, :].unsqueeze(0)
        # Wait only for this event's completion
        TotalMeasurement.frame_preproc_viewport_extraction.end_of_loop()
        return img_input
