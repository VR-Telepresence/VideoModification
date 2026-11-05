import threading
from dataclasses import dataclass
from datetime import datetime
from queue import  Queue

import cv2
import numpy as np
import torch

from time_measuring import TotalMeasurement


@dataclass(frozen=True, slots=True)
class ImageJob:
    img:torch.Tensor
    pos:torch.Tensor

class ParallaxWorker:
    def __init__(self, handle_parallax_image, model, width, height):
        self.__handle_parallax_image = handle_parallax_image
        self.__imgs = Queue(1)
        self.__model = model

        self._cpu_frame = torch.empty(
            (height, width, 3),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True
        )

        self._thread = threading.Thread(
            target=self._run_worker,
            name="parallax_worker",
            daemon=True,
        )
        self._thread.start()

    def queue_img(self, img, pos):
        if self.__imgs.full():
            self.__imgs.get()
            print("parallax worker queue is full, removed oldest image")
        self.__imgs.put(ImageJob(img, pos))




    def _run_worker(self):
        while True:
            img_job = self.__imgs.get()
            TotalMeasurement.frame_parallax_creation.start_of_loop()
            img = img_job.img
            pos = img_job.pos
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

            self.__handle_parallax_image(predicted_frame)
            TotalMeasurement.frame_parallax_creation.end_of_loop()
