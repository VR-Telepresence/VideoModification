import datetime
import logging
import threading
from dataclasses import dataclass
from queue import Queue
from typing import Any, Callable
import cv2
import numpy as np
import torch
from pytorch360convert import e2p
from typing_extensions import override

import parameters as params

from video_parallax.performance_tracking.time_measuring import TotalMeasurement
from video_parallax.worker.base_worker import BaseWorker, Job
from predict_view import getPositionVector
from video_parallax.worker.parallax_worker import ParallaxJob, ParallaxWorker
from video_parallax.worker.postprocess_worker import PostprocessWorker
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


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class PreprocessWorker(BaseWorker[PreprocessJob]):
    def __init__(self, pose_dims, width, height, handle_result_fn:Callable[[PreprocessResult],None]):
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
        self._stream = torch.cuda.Stream(
            device=params.DEVICE
        )


    @override
    def _handle_job(self, job: PreprocessJob):
        frame = job.frame
        frame_nr = job.frame_nr
        TotalMeasurement.frame_preproc.start_of_loop()
        TotalMeasurement.frame_is_read(frame_nr)

        # create example positions
        x = self.radius * np.cos(self.theta)
        y = self.radius * np.sin(self.theta)
        self.theta += np.pi / 50
        self.yaw += 2

        # create parallax
        pos = {
            "x": x,
            "y": y,
            "z": 0,
        }
        # frame = frame[120:120 + height, 200:200 + width, :]

        rotation = self.pose_server.camera_rotation()

        with torch.cuda.stream(self._stream):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(self._stream)

            viewport = self.extract_viewport(frame, rotation.yaw, -rotation.pitch)
            self.pass_frame_to_parallax_worker(viewport, pos, self.pose_dims, frame_nr)

            end.record(self._stream)
            TotalMeasurement.cuda_job_tracking.add_cuda_measurement_preproc(job.frame_nr, start, end)


        TotalMeasurement.frame_is_preproccessed(frame_nr)
        TotalMeasurement.frame_preproc.end_of_loop()

    def pass_frame_to_parallax_worker(self, frame, pos, pose_dims, frame_nr):
        # Preprocess Input
        img_input = frame[[2, 1, 0], :, :].unsqueeze(0)
        pos3D = self.pose_server.position_3d()
        print(pos3D.x, pos3D.y, pos3D.z)
        max_radius = 0.07
        x = max(min(-pos3D.x, max_radius), -max_radius)
        y = max(min(pos3D.y, max_radius), -max_radius)
        z = max(min(-pos3D.z, max_radius), -max_radius)

        pos = getPositionVector(x, y, z, 0, 0, 0, pose_dims=self.pose_dims).unsqueeze(
            0).to(params.DEVICE)
        ready_event = torch.cuda.Event()
        ready_event.record(self._stream)
        self.__handle_result_fn(PreprocessResult(img_input, pos, frame_nr, ready_event))


    def extract_viewport(self, frame, yaw, pitch=0, roll=0):

        logger.debug(f"""
            {threading.current_thread().name},
            "CUDA stream:",
            {torch.cuda.current_stream().cuda_stream}"""
        )

        frame_tensor = (
                torch.from_numpy(frame)
                .to(params.DEVICE)
                .permute(2, 0, 1)  # HWC -> CHW
                .float()
                / 255.0
        )

        viewport_tensor = e2p(
            frame_tensor,
            fov_deg=(90, 60), #horizontal, vertical
            h_deg=yaw,  # Yaw: center
            v_deg=pitch,  # Pitch: center
            out_hw=(self.height, self.width),
            mode="bilinear",
        )

        # Wait only for this event's completion
        return viewport_tensor
