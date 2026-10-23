def create_detector(name="yolo", **kw):
    if name == "yolo":
        from .detector_yolo import YoloDetector
        return YoloDetector(**kw)
    raise ValueError(f"Unknown modell: {name}")