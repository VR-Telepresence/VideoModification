from dataclasses import dataclass
from queue import Queue
from typing import Any
import cv2
import numpy as np
import torch
from pytorch360convert import e2p
from typing_extensions import override

import parameters as params

from video_parallax.time_measuring import TotalMeasurement
from video_parallax.worker.base_worker import BaseWorker, Job
from predict_view import getPositionVector
from video_parallax.worker.parallax_worker import ParallaxJob, ParallaxWorker
from video_parallax.worker.postprocess_worker import PostprocessWorker
from video_parallax.server.posetracker_server import PoseServer


@dataclass(frozen=True, slots=True)
class PreprocessJob(Job):
    frame:Any
    frame_nr: int


class PreprocessWorker(BaseWorker[PreprocessJob]):
    def __init__(self, pose_dims, parallax_model, detector, width, height):
        super().__init__(self.__class__.__name__, Queue(1))
        self.pose_server = PoseServer()
        self.postprocess_worker = PostprocessWorker(detector, width, height)
        self.parallax_worker = ParallaxWorker(self.postprocess_worker.add_job, parallax_model, width, height)

        self.width = width
        self.height = height


        self.theta = 0
        self.radius = 0.06
        self.yaw = 0

        self.pose_dims = pose_dims


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
        viewport = self.extract_viewport(frame, rotation.yaw, -rotation.pitch)
        self.pass_frame_to_parallax_worker(viewport, pos, self.pose_dims, frame_nr)
        TotalMeasurement.frame_is_preproccessed(frame_nr)
        TotalMeasurement.frame_preproc.end_of_loop()

    def pass_frame_to_parallax_worker(self, frame, pos, pose_dims, frame_nr):
        # Preprocess Input

        img_input = frame[[2, 1, 0], :, :].unsqueeze(0)
        pos3D = self.pose_server.position_3d()
        print(pos3D.x, pos3D.y, pos3D.z)

        pos = getPositionVector(pos["x"], pos["y"], pos["z"], 0, 0, 0, pose_dims=self.pose_dims).unsqueeze(
            0).to(params.DEVICE)

        self.parallax_worker.add_job(ParallaxJob(img_input, pos, frame_nr))

    def extract_viewport(self, frame, yaw, pitch=0, roll=0):
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

        return viewport_tensor
