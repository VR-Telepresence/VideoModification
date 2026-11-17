"""
Runs on the SMALL PC next to the camera.
Grabs the full 360 frame from the camera and sends it over NDI.
pip install opencv-python numpy ndi-python   (and install the NDI runtime)
"""
import cv2
import time
import socket
import struct
import threading
import NDIlib as ndi

#for insta x4 = 2880, 1440, 30
#for webcam = 1280, 720, 30
WIDTH, HEIGHT, FPS = 1280, 720, 30
NDI_NAME = "Insta360 X4"
SYNC_PORT = 5005   # UDP port for clock sync

def time_server():
    # answers every UDP ping with this PC's clock (nanoseconds)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("", SYNC_PORT))
    while True:
        _, addr = s.recvfrom(64)
        s.sendto(struct.pack("!q", time.time_ns()), addr)


threading.Thread(target=time_server, daemon=True).start()

def find_cap():
    # Try camera indexes 0-9, accept the first one that delivers the wanted size
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera {i}: {w}x{h}")
        if (w, h) == (WIDTH, HEIGHT):
            return cap
        cap.release()
    return None


def main():
    cap = find_cap()
    if cap is None:
        raise RuntimeError(f"No camera found that delivers {WIDTH}x{HEIGHT}")

    if not ndi.initialize():
        cap.release()
        raise RuntimeError("Could not initialize NDI (is the NDI runtime installed?)")

    settings = ndi.SendCreate()
    settings.ndi_name = NDI_NAME
    sender = ndi.send_create(settings)
    if sender is None:
        cap.release()
        raise RuntimeError("Could not create NDI sender")

    video = ndi.VideoFrameV2()
    video.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
    video.frame_rate_N = FPS * 1000
    video.frame_rate_D = 1000

    print(f"Sending NDI source '{NDI_NAME}'. Press Ctrl+C to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed")
                break
            video.timecode = time.time_ns() // 100
            video.data = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            ndi.send_send_video_v2(sender, video)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        ndi.send_destroy(sender)
        ndi.destroy()


if __name__ == "__main__":
    main()