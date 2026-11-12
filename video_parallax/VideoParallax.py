import datetime
import logging
import queue

import cv2


import numpy as np
import torch
from pytorch360convert import e2p


import parameters as params

from video_parallax.output_worker import PostprocessWorker
from video_parallax.parallax_worker import ParallaxWorker
from predict_view import getPositionVector

from video_parallax.server.posetracker_server import PoseServer
from video_parallax.spout_sender import UnitySpoutSender
from video_parallax.time_measuring import TotalMeasurement


class VideoParallax:
    height = 256
    width = 256

    def __init__(self, parallax_model, detector):
        width = self.width
        height = self.height
        self.pose_server = PoseServer()
        self.postprocess_worker = PostprocessWorker(detector, width, height)
        self.parallax_worker = ParallaxWorker(self.postprocess_worker.add_result_image, parallax_model, width, height)
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

    def create_parallax_with_cam(self, args, pose_dims ):
        cap = self.findCap()

        if cap is None:
            raise RuntimeError(
                "Could not find camera, check access of the system to the camera.")
        theta = 0
        radius = 0.06
        yaw=0


        frame_nr = 1
        while cap.isOpened():
            TotalMeasurement.frame_reading.start_of_loop()
            ret, frame = cap.read()
            TotalMeasurement.frame_is_read(frame_nr)

            # create example positions
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            theta+=np.pi/50
            yaw+=2

            # create parallax
            pos = {
                "x":x,
                "y":y,
                "z":0,
            }
            #frame = frame[120:120 + height, 200:200 + width, :]

            rotation = self.pose_server.camera_rotation()
            viewport = self.extract_viewport(frame, rotation.yaw, -rotation.pitch)
            self.pass_frame_to_parallax_worker(viewport, args, pos, pose_dims, frame_nr)
            TotalMeasurement.frame_is_preproccessed(frame_nr)
            TotalMeasurement.frame_reading.end_of_loop()
            display_frame = self.postprocess_worker.display_frame
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

    def pass_frame_to_parallax_worker(self, frame, args, pos, pose_dims, frame_nr):
        # Preprocess Input

        img_input = frame[[2, 1, 0], :, :].unsqueeze(0)
        pos3D = self.pose_server.position_3d()
        print(pos3D.x, pos3D.y, pos3D.z)

        pos = getPositionVector(pos["x"], pos["y"], pos["z"], 0, 0, 0, pose_dims=pose_dims).unsqueeze(
            0).to(params.DEVICE)

        self.parallax_worker.queue_img(img_input, pos, frame_nr)
