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
from VideoParallax import VideoParallax
from models.pvsnet_model import PVSNet
import helperFunctions as helper
from output_worker import PostprocessWorker
from parallax_worker import ParallaxWorker
from predict_view import getPositionVector
import pytorch360convert
from detection import create_detector

from server.posetracker_server import PoseServer
from server.viewport_providing_server import ViewportProvidingServer
from spout_sender import UnitySpoutSender
from time_measuring import TotalMeasurement

def build_detector(args):
    if args.detector == "none":
        return None
    only = tuple(s.strip() for s in args.only.split(","))
    common = dict(every_n=args.det_every, style=args.style, draw_boxes=args.boxes)
    if args.detector == "yolo":
        return create_detector("yolo", only=only, conf=args.det_conf,
                               imgsz=args.det_imgsz, **common)

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
    VideoParallax(model, detector).create_parallax_with_cam(args, pose_dims)