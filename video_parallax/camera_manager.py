import logging

import NDIlib as ndi
import numpy as np
import cv2

from video_parallax.performance_tracking.time_measuring import TotalMeasurement


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

class CameraManager:
    def __init__(self):
        self.cap = self.__findCap()
        self.logger = logging.Logger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())
        pass

    def __findCap(self):
        # NDI first, USB camera as fallback
        cap = self.__findNdiCap() or self.__findUsbCap()
        if cap is None:
            raise RuntimeError(
                "Could not find camera, check access of the system to the camera.")
        return cap

    def __findNdiCap(self):
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
        print("Using NDI")
        return NDICapture(receiver)

    def __findUsbCap(self):
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

            if cap.isOpened():
                print("Using USB")
                return cap
            cap.release()
        return None


    def get_next_frame(self):
        if not self.cap.isOpened():
            self.cap = self.__findCap()

        while True:
            TotalMeasurement.frame_reading.start_of_loop()
            ret, frame = self.cap.read()
            TotalMeasurement.frame_reading.end_of_loop()

            if not ret or frame is None:
                self.logger.info("No frame received, waiting for next frame.")
                continue

            return frame

    def stop_camera(self):
        self.cap.release()

