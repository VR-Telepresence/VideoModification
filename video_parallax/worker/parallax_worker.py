from dataclasses import dataclass
from queue import  Queue
from typing import override

import torch
from video_parallax.worker.base_worker import BaseWorker
from video_parallax.time_measuring import TotalMeasurement
from video_parallax.worker.base_worker import Job
from video_parallax.worker.output_worker import PostprocessJob


@dataclass(frozen=True, slots=True)
class ParallaxJob(Job):
    img:torch.Tensor
    pos:torch.Tensor
    frame_nr:int

class ParallaxWorker(BaseWorker[ParallaxJob]):
    def __init__(self, handle_parallax_image, model, width, height):
        super().__init__(self.__class__.__name__, Queue(1))
        self.__handle_parallax_image = handle_parallax_image
        self.__model = model

        self._cpu_frame = torch.empty(
            (height, width, 3),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True
        )

    @override
    def _handle_job(self, job: ParallaxJob):
        TotalMeasurement.frame_parallax_creation.start_of_loop()
        img = job.img
        pos = job.pos
        with torch.inference_mode():
            predicted_img = self.__model(img, pos)
            # Model output: [1, 3, H, W], Float32 RGB, CUDA
            frame_gpu = (
                predicted_img[0]  # [3, H, W]
                .clamp(0, 1)
                .mul(255)
                .to(torch.uint8)  # UInt8 on GPU
                .permute(1, 2, 0)  # [H, W, 3]
                .flip(-1)  # RGB -> BGR
                .contiguous()
            )

        # Transfer only the final UInt8 frame, saving around 0.5ms of 57ms total
        self._cpu_frame.copy_(frame_gpu, non_blocking=True)

        # Wait until the transfer has finished
        torch.cuda.current_stream().synchronize()
        predicted_frame = self._cpu_frame.numpy()

        self.__handle_parallax_image(PostprocessJob(predicted_frame, job.frame_nr))

        TotalMeasurement.frame_parallax_is_created(job.frame_nr)
        TotalMeasurement.frame_parallax_creation.end_of_loop()