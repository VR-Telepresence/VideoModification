import logging
import threading
from dataclasses import dataclass
from queue import  Queue
from typing import override, Any, Callable

import torch
from video_parallax.worker.base_worker import BaseWorker
from video_parallax.performance_tracking.time_measuring import TotalMeasurement
from video_parallax.worker.base_worker import Job
from video_parallax.worker.postprocess_worker import PostprocessJob, PostprocessWorker
import parameters as params


@dataclass(frozen=True, slots=True)
class ParallaxJob(Job):
    img:torch.Tensor
    pos:torch.Tensor
    frame_nr:int
    ready_event:torch.Event

@dataclass(frozen=True, slots=True)
class ParallaxResult(Job):
    frame:Any
    frame_nr: int

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ParallaxWorker(BaseWorker[ParallaxJob]):
    def __init__(self, model, width, height, handle_result_fn:Callable[[ParallaxResult],None]):
        super().__init__(self.__class__.__name__, Queue(1))
        self.__model = model
        self.__handle_result_fn = handle_result_fn

        self._cpu_frame = torch.empty(
            (height, width, 3),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=torch.cuda.is_available()
        )
        if params.DEVICE == "cuda":
            self._stream = torch.cuda.Stream(device="cuda")
        else:
            self._stream = None

    def __do_inference_gpu(self, job: ParallaxJob):

        with torch.cuda.stream(self._stream):
            logger.debug(f"""
                        {threading.current_thread().name},
                        "CUDA stream:",
                        {torch.cuda.current_stream().cuda_stream}"""
                         )

            self._stream.wait_event(job.ready_event)

            # Track input tensors' use on this stream
            job.img.record_stream(self._stream)
            job.pos.record_stream(self._stream)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(self._stream)

            self.__do_inference(job)

            end.record(self._stream)

            TotalMeasurement.cuda_job_tracking.add_cuda_measurement_parallax(job.frame_nr, start, end)

        TotalMeasurement.frame_parallax_is_created(job.frame_nr)
        TotalMeasurement.frame_parallax_creation.end_of_loop()

    @override
    def _handle_job(self, job: ParallaxJob):
        if params.DEVICE == "cpu":
            self.__do_inference(job)
        elif params.DEVICE == "cuda":
            self.__do_inference_gpu(job)
        else:
            raise NotImplementedError("Invalid device value")

    def __do_inference(self, job: ParallaxJob):
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
        if frame_gpu.is_cuda:
            torch.cuda.current_stream().synchronize()

        predicted_frame = self._cpu_frame.numpy()
        self.__handle_result_fn(ParallaxResult(predicted_frame, job.frame_nr))