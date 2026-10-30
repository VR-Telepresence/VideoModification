import threading
from dataclasses import dataclass
from queue import  Queue

import cv2
import numpy as np
import torch



@dataclass(frozen=True, slots=True)
class ImageJob:
    img:torch.Tensor
    pos:torch.Tensor

class ParallaxWorker:
    def __init__(self, res_queue: Queue, model):
        self.__res_imgs = res_queue
        self.__imgs = Queue(5)
        self.__model = model

        self._thread = threading.Thread(
            target=self._run_worker,
            name="parallax_worker",
            daemon=True,
        )
        self._thread.start()

    def queue_img(self, img, pos):
        self.__imgs.put(ImageJob(img, pos))


    def _run_worker(self):
        while True:
            print("parrallax")

            img_job = self.__imgs.get()
            img = img_job.img
            pos = img_job.pos
            with torch.no_grad():
                predicted_img = self.__model(img, pos)
                predicted_img = torch.clamp(predicted_img, 0, 1)
            #torch.cuda.synchronize() # can be removed, only for time tracking relevant


            predicted_np = predicted_img[0].squeeze(0).permute(1, 2, 0).cpu().numpy()
            # Float [0,1] -> uint8 [0,255]
            predicted_np = (predicted_np * 255).astype(np.uint8)
            # RGB -> BGR
            predicted_frame = cv2.cvtColor(predicted_np, cv2.COLOR_RGB2BGR)

            self.__res_imgs.put(predicted_frame)
