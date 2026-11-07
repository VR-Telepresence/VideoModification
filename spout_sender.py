import cv2
import numpy as np
import time
import SpoutGL

try:
    from SpoutGL.enums import GL_BGRA_EXT
except ImportError:
    from OpenGL.GL import GL_BGRA as GL_BGRA_EXT


class UnitySpoutSender:
    """
    Sends OpenCV-BGR-Frames via Spout to Unity.
    """

    def __init__(self, sender_name="PVSNetStream"): # must match the name in the Unity spout receiver component
        self.sender_name = sender_name
        self.sender = SpoutGL.SpoutSender()
        self.sender.setSenderName(sender_name)

        self.bgra_buffer = None
        self.width = 0
        self.height = 0

        print(f"[Spout] Sender started: {self.sender_name}")

    def _ensure_buffer(self, width, height):
        if (
            self.bgra_buffer is not None
            and self.width == width
            and self.height == height
        ):
            return

        self.width = width
        self.height = height
        self.bgra_buffer = np.empty(
            (height, width, 4),
            dtype=np.uint8
        )
        print(f"[Spout] Image buffer created: {width}x{height}, BGRA8")

    def send_bgr(self, frame_bgr):
        if frame_bgr is None:
            return False

        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("frame_bgr must be a NumPy-Array")

        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be  H x W x 3")

        if frame_bgr.dtype != np.uint8:
            frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)

        if not frame_bgr.flags.c_contiguous:
            frame_bgr = np.ascontiguousarray(frame_bgr)

        height, width = frame_bgr.shape[:2]
        self._ensure_buffer(width, height)

        # convert BGR to BGRA and write to buffer
        cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2BGRA,
            dst=self.bgra_buffer
        )

        # send to Spout
        success = self.sender.sendImage(
            self.bgra_buffer,
            width,
            height,
            GL_BGRA_EXT,
            False,
            0
        )

        return success

    def close(self):
        if self.sender:
            self.sender.releaseSender()
            print("[Spout] Sender closed.")