import datetime
import logging
import queue
import time

import cv2
import os
import argparse

import numpy as np
import torch
from pytorch360convert import e2p
from scipy.spatial.transform import rotation
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import parameters as params
from models.pvsnet_model import PVSNet
import helperFunctions as helper
from output_worker import OutputWorker
from parallax_worker import ParallaxWorker
from predict_view import getPositionVector
import pytorch360convert

from server.posetracker_server import PoseServer
from server.viewport_providing_server import ViewportProvidingServer
from time_measuring import TotalMeasurement


class VideoParallax:
    height = 256
    width = 256

    def __init__(self, parallax_model, args):
        self.pose_server = PoseServer()

        only = tuple(
            s.strip()
            for s in args.only.split(",")
            if s.strip()
        )

        self.unity_server = ViewportProvidingServer(
            detector=args.detector,
            classes=only,
            conf=args.det_conf,
            imgsz=args.det_imgsz,
        )

        self.output_worker = OutputWorker(self.unity_server)
        self.parallax_worker = ParallaxWorker(self.output_worker.add_result_image, parallax_model)

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
                .permute(2, 0, 1)  # HWC -> CHW
                .float()
                / 255.0
        )

        viewport_tensor = e2p(
            frame_tensor,
            fov_deg=(90, 60), #horizontal, vertical
            h_deg=yaw,  # Yaw: center
            v_deg=pitch,  # Pitch: center
            out_hw=(height, width),
            mode="bilinear",
        )

        viewport = (
            viewport_tensor
            .detach()
            .permute(1, 2, 0)  # CHW -> HWC
            .clamp(0, 1)
            .mul(255)
            .byte()
            .cpu()
            .numpy()
        )
        return viewport

    def create_parallax_with_cam(self,  args, pose_dims):
        cap = self.findCap()

        if cap is None:
            raise RuntimeError(
                "Could not find camera, check access of the system to the camera.")
        theta = 0
        radius = 0.06
        yaw=0

        while cap.isOpened():
            TotalMeasurement.frame_reading.start_of_loop()
            ret, frame = cap.read()

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
            TotalMeasurement.frame_reading.end_of_loop()
            rotation = self.pose_server.camera_rotation()
            viewport = self.extract_viewport(frame, rotation.yaw, rotation.pitch)
            self.pass_frame_to_parallax_worker(viewport, args, pos, pose_dims)

            display_frame = self.output_worker.display_frame
            if display_frame is not None:
                cv2.imshow("Parallax", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            TotalMeasurement.print_summary()

        # Close the image show frame
        cv2.destroyAllWindows()
        self.writer.release()

    def pass_frame_to_parallax_worker(self, frame, args, pos, pose_dims):
        # Preprocess Input

        transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor()
        ])

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_input = transform(
            Image.fromarray(frame_rgb)
        ).unsqueeze(0).to(params.DEVICE)
        # --- 0.001 s up

        pos = getPositionVector(pos["x"], pos["y"], pos["z"], 0, 0, 0, pose_dims=pose_dims).unsqueeze(
            0).to(params.DEVICE)

        self.parallax_worker.queue_img(img_input, pos)

        #0.01s up
        #t0 = time.perf_counter()
        #(1, 3, H, W) -> (H, W, 3)
        #predicted_np = predicted_img[0].squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Float [0,1] -> uint8 [0,255]
        #predicted_np = (predicted_np * 255).astype(np.uint8)
        # RGB -> BGR
        #predicted_frame = cv2.cvtColor(predicted_np, cv2.COLOR_RGB2BGR)
        #t1 = time.perf_counter()

        #print(f"Postprocessing: {(t1 - t0) * 1000:.3f} ms")
        #return predicted_frame

if __name__ == "__main__":

    print("PyTorch:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Compute capability:", torch.cuda.get_device_capability(0))
        print("Supported architectures:", torch.cuda.get_arch_list())


    parser = argparse.ArgumentParser(description="Predict a single novel view using PVSNet")
    parser.add_argument("--output", type=str, default="output_view.png", help="Output image path")
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/checkpoint_coco_pvsnet_lite_256x256.pth",
                        help="Model checkpoint path")
    parser.add_argument("--dataset", type=str, choices=["blender", "coco"], default="coco",
                        help="Dataset type to determine pose dimensions (blender=3, coco=6)")

    parser.add_argument("--detector", choices=["none", "yolo"], default="none")
    parser.add_argument("--only", type=str, default="person", help='z.B. "vase" oder "person,cup"')
    parser.add_argument("--det_conf", type=float, default=0.25)
    parser.add_argument("--det_imgsz", type=int, default=256)

    args = parser.parse_args()

    # Parse config from checkpoint name or arguments
    base_name = os.path.basename(args.checkpoint).replace(".pth", "")
    is_lite = "lite" in base_name

    if "256x256" in base_name:
        height, width = 256, 256
    elif "512x512" in base_name:
        height, width = 512, 512
    else:
        height, width = 256, 256


    pose_dims = 3 if  "blender" in args.checkpoint.lower() else 6

    print(f"Status: Loading Model (Resolution: {height}x{width}, Lite: {is_lite}, Pose Dims: {pose_dims})")

    model = PVSNet(total_image_input=params.params_number_input, pose_dims=pose_dims, height=height, width=width,
                   is_lite=is_lite)
    model = helper.load_Checkpoint(args.checkpoint, model, load_cpu=True)
    model.to(params.DEVICE)
    model.eval()

    VideoParallax(model, args).create_parallax_with_cam(args, pose_dims)