import av
import cv2

# pip install av opencv-python

SENDER_IP = "0.0.0.1"   # need to fill
PORT = 9000

container = av.open(
    f"srt://{SENDER_IP}:{PORT}?mode=caller&latency=150000",
    options={"fflags": "nobuffer", "flags": "low_delay"},
)

try:
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="bgr24")
        cv2.imshow("SRT", cv2.resize(img, None, fx=0.4, fy=0.4))
        if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
            break
except KeyboardInterrupt:
    pass
finally:
    container.close()
    cv2.destroyAllWindows()