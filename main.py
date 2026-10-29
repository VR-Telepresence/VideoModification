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
from posetracker_server import PoseServer
from predict_view import getPositionVector
import pytorch360convert
from detection import create_detector


class VideoParallax:
    height = 256
    width = 256

    def __init__(self):
        self.pose_server = PoseServer()



    def create_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30
        return  cv2.VideoWriter(
            "viewport.mp4",
            fourcc,
            fps,
            (width, height)
        )


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



    def create_parallax_with_cam(self,  args, pose_dims, detector=None ):
        cap = self.findCap()

        if cap is None:
            raise RuntimeError(
                "Could not find camera, check access of the system to the camera.")

        theta = 0
        radius = 0.06
        yaw=0
        writer = self.create_writer()

        while cap.isOpened():
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
            rotation = self.pose_server.camera_rotation()
            viewport = self.extract_viewport(frame, rotation.yaw, rotation.pitch)

            viewport = self.create_parallax( viewport, args, pos, pose_dims)

            if detector is not None:
                detector.submit(viewport)
                viewport = detector.draw(viewport)

            # show parallax to user
            cv2.imshow("", viewport)
            writer.write(viewport)
            # wait for key
            key_val = cv2.waitKey(10) & 0XFF

            # break gracefully
            if key_val == ord('q'):
                break
        # Close the image show frame
        cv2.destroyAllWindows()
        writer.release()



    def create_parallax(self, frame, args, pos, pose_dims):
        # Preprocess Input
        transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor()
        ])

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_input = transform(
            Image.fromarray(frame_rgb)
        ).unsqueeze(0).to(params.DEVICE)

        pos = getPositionVector(pos["x"], pos["y"], pos["z"], 0, 0, 0, pose_dims=pose_dims).unsqueeze(
            0).to(params.DEVICE)

        # Inference
        with torch.no_grad():
            predicted_img = model(img_input, pos)
            predicted_img = torch.clamp(predicted_img, 0, 1)

        # (1, 3, H, W) -> (H, W, 3)
        predicted_np = predicted_img[0].squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Float [0,1] -> uint8 [0,255]
        predicted_np = (predicted_np * 255).astype(np.uint8)

        # RGB -> BGR
        predicted_frame = cv2.cvtColor(predicted_np, cv2.COLOR_RGB2BGR)

        return predicted_frame

def build_detector(args):
    if args.detector == "none":
        return None
    only = tuple(s.strip() for s in args.only.split(","))
    common = dict(every_n=args.det_every, style=args.style, draw_boxes=args.boxes)
    if args.detector == "yolo":
        return create_detector("yolo", only=only, conf=args.det_conf,
                               imgsz=args.det_imgsz, **common)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a single novel view using PVSNet")
    parser.add_argument("--output", type=str, default="output_view.png", help="Output image path")
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/checkpoint_coco_pvsnet_lite_256x256.pth",
                        help="Model checkpoint path")
    parser.add_argument("--dataset", type=str, choices=["blender", "coco"], default="coco",
                        help="Dataset type to determine pose dimensions (blender=3, coco=6)")

    parser.add_argument("--detector", choices=["none", "yolo", "sam3"], default="none")
    parser.add_argument("--only", type=str, default="person", help='z.B. "vase" oder "person,cup"')
    parser.add_argument("--det_conf", type=float, default=0.25)
    parser.add_argument("--det_imgsz", type=int, default=320)
    parser.add_argument("--det_every", type=int, default=2)
    parser.add_argument("--style", choices=["outline", "fill"], default="outline")
    parser.add_argument("--boxes", action="store_true")

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

    detector = build_detector(args)

    VideoParallax().create_parallax_with_cam(args, pose_dims, detector)


## start with
# python main.py  --checkpoint .\checkpoint\checkpoint_blender_pvsnet_256x256.pth


# Dependencies
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# pip install gradio , opencv-python, random-fourier-features-pytorch, pytorch360convert, ultralytics