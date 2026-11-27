import sys
import time
import traceback
from typing import Final

import cv2
import numpy as np
import NDIlib as ndi

from video_parallax.camera_manager import CameraManager
from video_parallax.detection import create_detector
from video_parallax.pipeline import Pipeline
from video_parallax.performance_tracking.time_measuring import TotalMeasurement

sys.setswitchinterval(0.0005)

def build_detector(args):
    if args.detector == "none":
        return None
    only = tuple(s.strip() for s in args.only.split(","))
    return create_detector(args.detector, only=only, conf=args.det_conf, imgsz=args.det_imgsz)



class VideoParallax:

    def __init__(self, parallax_model, args, pose_dims, width, height):
        detector = build_detector(args)
        self.pipeline = Pipeline(pose_dims, parallax_model, detector, width, height)
        self.camera_manager = CameraManager()

    def video_parallax_program_loop(self):
        while True:
            try:
                self.create_parallax_with_cam()
            except Exception as e:
                traceback.print_exc()
                time.sleep(1)

    def create_parallax_with_cam(self):
        self.pipeline.start()
        frame_nr = 1
        try:
            while True:
                frame = self.camera_manager.get_next_frame()
                self.pipeline.queue_job(frame, frame_nr)

                display_frame = self.pipeline.display_frame
                if display_frame is not None:
                    cv2.imshow("Parallax", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                TotalMeasurement.write_summary()
                frame_nr += 1
        finally:
            self.camera_manager.stop_camera()
            self.pipeline.stop()
            cv2.destroyAllWindows()
            if self.pipeline.spout_sender is not None:
                self.pipeline.spout_sender.close()

