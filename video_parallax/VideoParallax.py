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

    def __init__(self, parallax_model, detector, use_ndi):
        self.use_ndi = use_ndi
        width = self.width
        height = self.height
        self.pose_server = PoseServer()
        self.postprocess_worker = PostprocessWorker(detector, width, height)
        self.parallax_worker = ParallaxWorker(self.postprocess_worker.add_result_image, parallax_model, width, height)
        self.spout_sender = UnitySpoutSender("PVSNetStream")
        # """

    class NDICam:
        def __init__(self):
            import NDIlib as ndi  # only needed when NDI is used
            self.ndi = ndi
            self.finder = None
            self.receiver = None

            ndi.initialize()
            self.finder = ndi.find_create_v2()
            sources = []
            for _ in range(5):
                ndi.find_wait_for_sources(self.finder, 2000)
                sources = ndi.find_get_current_sources(self.finder)
                if sources:
                    break
            if not sources:
                return

            print("Connecting to", sources[0].ndi_name)
            settings = ndi.RecvCreateV3()
            settings.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
            self.receiver = ndi.recv_create_v3(settings)
            ndi.recv_connect(self.receiver, sources[0])

        def isOpened(self):
            return self.receiver is not None

        def read(self):
            frame_type, video, _a, _m = self.ndi.recv_capture_v2(self.receiver, 1000)
            if frame_type != self.ndi.FRAME_TYPE_VIDEO:
                return False, None
            frame = np.copy(video.data)
            self.ndi.recv_free_video_v2(self.receiver, video)
            return True, cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        def release(self):
            if self.receiver is not None:
                self.ndi.recv_destroy(self.receiver)
                self.receiver = None
            if self.finder is not None:
                self.ndi.find_destroy(self.finder)
                self.finder = None
            self.ndi.destroy()
        # """

    def findCap(self):
        if self.use_ndi:
            cap = self.NDICam()
            if cap.isOpened():
                return cap
            cap.release()
            print("Failed to find NDI source.")
            return None

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
            if cap.isOpened() and i ==4:
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
            TotalMeasurement.frame_reading.end_of_loop()
            TotalMeasurement.frame_preproc.start_of_loop()
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
            TotalMeasurement.frame_preproc.end_of_loop()
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
