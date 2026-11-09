#pip install opencv-python
#FFmpeg must be installed and on PATH (with SRT support).
import shutil
import subprocess

import cv2

#for insta x4 = 2880, 1440, 30
#for webcam = 1280, 720, 30

WIDTH, HEIGHT, FPS = 2880, 1440, 30
BITRATE = "25M"
ENCODER = "libx264"  # hardware: h264_nvenc (NVIDIA), h264_qsv (Intel), h264_amf (AMD)
PORT = 9000

if shutil.which("ffmpeg") is None:
    raise SystemExit("ffmpeg not found. Install it or add it to PATH.")


def find_cap():
    # Take the first camera that really delivers the full 360 frame
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


cap = find_cap()
if cap is None:
    raise SystemExit(f"No camera found that delivers {WIDTH}x{HEIGHT}")

# Only libx264 and h264_nvenc understand these low-latency presets
if ENCODER == "libx264":
    preset = ["-preset", "ultrafast", "-tune", "zerolatency"]
elif ENCODER == "h264_nvenc":
    preset = ["-preset", "p1", "-tune", "ll"]
else:
    preset = []

ffmpeg = subprocess.Popen([
    "ffmpeg", "-loglevel", "warning",
    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
    "-c:v", ENCODER, *preset, "-pix_fmt", "yuv420p",
    "-b:v", BITRATE, "-g", str(FPS), "-bf", "0",
    "-f", "mpegts", f"srt://0.0.0.0:{PORT}?mode=listener&latency=150000",
], stdin=subprocess.PIPE)

print(f"Streaming on srt://0.0.0.0:{PORT}. Waiting for the main PC. Ctrl+C to stop.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break
        ffmpeg.stdin.write(frame.tobytes())
except (KeyboardInterrupt, BrokenPipeError, OSError):
    pass
finally:
    cap.release()
    try:
        ffmpeg.stdin.close()
    except Exception:
        pass
    ffmpeg.wait()
