import datetime
import logging
import queue

import cv2


import numpy as np
import torch
from pytorch360convert import e2p


import parameters as params

from video_parallax.worker.preprocess_worker import PostprocessWorker, PreprocessWorker, PreprocessJob
from video_parallax.worker.parallax_worker import ParallaxWorker, ParallaxJob
from predict_view import getPositionVector

from video_parallax.server.posetracker_server import PoseServer
from video_parallax.spout_sender import UnitySpoutSender
from video_parallax.time_measuring import TotalMeasurement


class VideoParallax:
    height = 256
    width = 256

    def __init__(self, parallax_model, detector, args, pose_dims):
        width = self.width
        height = self.height
        self.preprocess_worker = PreprocessWorker(pose_dims, parallax_model, detector, width, height)
        self.spout_sender = UnitySpoutSender("PVSNetStream")




    def findCap(self):
        # Establish a connection to the webcam
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            # Videoformat setzen
            cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG")
            )

            # 360°-Panorama aktivieren
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2880)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # print(i, cap.isOpened())
            if cap.isOpened():
                return cap
            cap.release()
        print("Failed to find camera.")
        return None


    def create_parallax_with_cam(self):
        cap = self.findCap()

        if cap is None:
            raise RuntimeError(
                "Could not find camera, check access of the system to the camera.")


        frame_nr = 1
        while cap.isOpened():
            TotalMeasurement.frame_reading.start_of_loop()
            ret, frame = cap.read()
            TotalMeasurement.frame_reading.end_of_loop()

            self.preprocess_worker.add_job(PreprocessJob(frame, frame_nr))

            display_frame = self.preprocess_worker.postprocess_worker.display_frame
            if display_frame is not None:
                self.spout_sender.send_bgr(display_frame)
                cv2.imshow("Parallax", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            TotalMeasurement.write_summary()

            frame_nr += 1


        # Close the image show frame
        cv2.destroyAllWindows()
        self.spout_sender.close()

