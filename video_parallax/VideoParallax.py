from typing import Final

import cv2

from video_parallax.pipeline import Pipeline
from video_parallax.worker.preprocess_worker import PostprocessWorker, PreprocessWorker, PreprocessJob
from video_parallax.spout_sender import UnitySpoutSender
from video_parallax.performance_tracking.time_measuring import TotalMeasurement

class VideoParallax:

    height:Final[int] = 256
    width:Final[int] = 256

    def __init__(self, parallax_model, detector, pose_dims):
        width = self.width
        height = self.height
        self.pipeline = Pipeline(pose_dims, parallax_model, detector, width, height)


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


    def create_parallax_with_cam(self):
        self.pipeline.start()
        cap = self.findCap()

        if cap is None:
            raise RuntimeError(
                "Could not find camera, check access of the system to the camera.")

        frame_nr = 1
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

        # Close the image show frame
        cv2.destroyAllWindows()
        self.spout_sender.close()

