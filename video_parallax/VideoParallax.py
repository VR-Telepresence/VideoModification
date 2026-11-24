from typing import Final

import cv2
import numpy as np
import NDIlib as ndi

from video_parallax.detection import create_detector
from video_parallax.pipeline import Pipeline
from video_parallax.performance_tracking.time_measuring import TotalMeasurement

def build_detector(args):
    if args.detector == "none":
        return None
    only = tuple(s.strip() for s in args.only.split(","))
    return create_detector(args.detector, only=only, conf=args.det_conf, imgsz=args.det_imgsz)
class NDICapture:
    """Behaves like cv2.VideoCapture (isOpened/read/release) but reads from NDI."""

    def __init__(self, receiver):
        self.receiver = receiver
        self.opened = True

    def isOpened(self):
        return self.opened

    def read(self):
        frame_type, video, _audio, _meta = ndi.recv_capture_v2(self.receiver, 1000)
        if frame_type != ndi.FRAME_TYPE_VIDEO:
            return False, None
        frame = np.copy(video.data)  # copy before freeing NDI's buffer
        ndi.recv_free_video_v2(self.receiver, video)
        return True, cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def release(self):
        self.opened = False
        ndi.recv_destroy(self.receiver)
        ndi.destroy()

class VideoParallax:
    height:Final[int] = 256
    width:Final[int] = 256

    def __init__(self, parallax_model, args, pose_dims):
        width = self.width
        height = self.height

        detector = build_detector(args)
        self.pipeline = Pipeline(pose_dims, parallax_model, detector, width, height)

    def findNdiCap(self):
        # Connect to the first NDI source found on the network
        if ndi is None:
            return None
        ndi.initialize()
        finder = ndi.find_create_v2()
        sources = []
        for _ in range(3):  # ~6 seconds
            ndi.find_wait_for_sources(finder, 2000)
            sources = ndi.find_get_current_sources(finder)
            if sources:
                break
        if not sources:
            ndi.find_destroy(finder)
            ndi.destroy()
            return None

        print("Connecting to", sources[0].ndi_name)
        settings = ndi.RecvCreateV3()
        settings.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        receiver = ndi.recv_create_v3(settings)
        ndi.recv_connect(receiver, sources[0])
        ndi.find_destroy(finder)
        return NDICapture(receiver)


    def findUsbCap(self):
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

            if cap.isOpened() and i == 0:
                return cap
            cap.release()
        return None


    def findCap(self):
        # NDI first, USB camera as fallback
        cap = self.findNdiCap() or self.findUsbCap()
        if cap is None:
            print("Failed to find camera.")
        return cap


    def create_parallax_with_cam(self):
        self.pipeline.start()
        cap = self.findCap()

        if cap is None:
            raise RuntimeError(
                "Could not find camera, check access of the system to the camera.")

        frame_nr = 1
        try:
            while cap.isOpened():
                TotalMeasurement.frame_reading.start_of_loop()
                ret, frame = cap.read()
                TotalMeasurement.frame_reading.end_of_loop()

                if not ret or frame is None:
                    continue

                self.pipeline.queue_job(frame, frame_nr)

                display_frame = self.pipeline.display_frame
                if display_frame is not None:
                    cv2.imshow("Parallax", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                TotalMeasurement.write_summary()
                frame_nr += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.pipeline.spout_sender is not None:
                self.pipeline.spout_sender.close()

