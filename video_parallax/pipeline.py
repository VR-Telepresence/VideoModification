from typing import Any

from video_parallax.performance_tracking.time_measuring import TotalMeasurement
from video_parallax.spout_sender import UnitySpoutSender
from video_parallax.worker.postprocess_worker import PostprocessWorker, PostprocessJob, PostprocessResult
from video_parallax.worker.parallax_worker import ParallaxWorker, ParallaxJob, ParallaxResult
from video_parallax.worker.preprocess_worker import PreprocessWorker, PreprocessResult, PreprocessJob


class Pipeline:
    def __init__(self, pose_dims, parallax_model, detector, width, height):
        self.display_frame = None

        self.postprocess_worker = PostprocessWorker(detector, width, height, self._handle_final_result)
        self.parallax_worker = ParallaxWorker(parallax_model, width, height, self._to_postprocess)
        self.preprocess_worker = PreprocessWorker(pose_dims, width, height, self._to_parallax)

        self.spout_sender = UnitySpoutSender("PVSNetStream")





    def _to_parallax(self, result: PreprocessResult) -> None:
        job = ParallaxJob(
            img=result.img,
            pos=result.pos,
            frame_nr=result.frame_nr,
            ready_event=result.ready_event,
        )

        self.parallax_worker.add_job(job)

    def _to_postprocess(self, result: ParallaxResult) -> None:
        job = PostprocessJob(
            result.frame, result.frame_nr
        )
        self.postprocess_worker.add_job(job)

    def _handle_final_result(self, result: PostprocessResult) -> None:
        self.display_frame = result.frame
        self.spout_sender.send_bgr(result.frame)
        TotalMeasurement.frame_is_send(result.frame_nr)


    def start(self) -> None:
        self.postprocess_worker.start()
        self.parallax_worker.start()
        self.preprocess_worker.start()


    def queue_job(self, frame:Any, frame_nr:int) -> None:
        self.preprocess_worker.add_job(PreprocessJob(frame, frame_nr))