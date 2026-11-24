from typing import Final

import cv2

from video_parallax.detection import create_detector
from video_parallax.pipeline import Pipeline
from video_parallax.performance_tracking.time_measuring import TotalMeasurement

def build_detector(args):
    if args.detector == "none":
        return None
    only = tuple(s.strip() for s in args.only.split(","))
    return create_detector(args.detector, only=only, conf=args.det_conf, imgsz=args.det_imgsz)

class VideoParallax:
    height:Final[int] = 256
    width:Final[int] = 256

    def __init__(self, parallax_model, args, pose_dims, use_ndi):
        width = self.width
        height = self.height

        self.use_ndi = use_ndi

        detector = build_detector(args)
        self.pipeline = Pipeline(pose_dims, parallax_model, detector, width, height)

    class NDICam:
        def __init__(self):
            import NDIlib as ndi  # only needed when NDI is used
            self.ndi = ndi
            self.finder = None
            self.receiver = None

        detector = build_detector(args)
        self.pipeline = Pipeline(pose_dims, parallax_model, detector, width, height)
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
            if cap.isOpened():
                return cap
            cap.release()
        print("Failed to find camera.")
        return None


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

