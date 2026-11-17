import time
import socket
import struct
import cv2
import numpy as np
import NDIlib as ndi

# pip install opencv-python numpy ndi-python(and install the NDI runtime)

SYNC_PORT = 5005          # must match the sender
RESYNC_EVERY_S = 30       # clocks drift, so measure the offset again now and then

def get_offset_ns(ip, samples=30):
    # returns (rtt_ns, offset_ns) where offset = sender clock - receiver clock
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.5)
    best = None
    for _ in range(samples):
        t0 = time.time_ns()
        s.sendto(b"x", (ip, SYNC_PORT))
        try:
            data, _ = s.recvfrom(64)
        except socket.timeout:
            continue
        t1 = time.time_ns()
        (sender_ns,) = struct.unpack("!q", data)
        rtt = t1 - t0
        offset = sender_ns - (t0 + t1) // 2
        if best is None or rtt < best[0]:
            best = (rtt, offset)
    s.close()
    return best

ndi.initialize()

# Find an NDI source on the network
finder = ndi.find_create_v2()
sources = []
while not sources:
    ndi.find_wait_for_sources(finder, 2000)
    sources = ndi.find_get_current_sources(finder)
src = sources[0]
print("Connecting to", src.ndi_name)


sender_ip = src.url_address.split(":")[0]
result = get_offset_ns(sender_ip)
if result is None:
    raise RuntimeError("No reply from sender time server (firewall? sender running?)")
rtt, offset_ns = result
print(f"clock offset {offset_ns / 1e6:+.2f} ms (sync RTT {rtt / 1e6:.2f} ms)")
last_sync = time.monotonic()

# Connect and ask for BGRA frames
settings = ndi.RecvCreateV3()
settings.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
receiver = ndi.recv_create_v3(settings)
ndi.recv_connect(receiver, sources[0])
ndi.find_destroy(finder)

num = 0
try:
    while True:
        frame_type, video, _audio, _meta = ndi.recv_capture_v2(receiver, 1000)
        if frame_type != ndi.FRAME_TYPE_VIDEO:
            continue
        now_ns = time.time_ns()
        sent_ns = video.timecode * 100 - offset_ns  # sender stamp, in receiver clock

        frame = np.copy(video.data)               # copy before freeing NDI's buffer
        ndi.recv_free_video_v2(receiver, video)
        latency_ms = (now_ns - sent_ns) / 1e6

        num += 1
        if num % 30 == 0:  # print about once per second
            print(f"sent: {sent_ns / 1e9:.3f} s | received: {now_ns / 1e9:.3f} s | latency: {latency_ms:.1f} ms")

        if time.monotonic() - last_sync > RESYNC_EVERY_S:
            r = get_offset_ns(sender_ip, samples=10)
            if r is not None:
                offset_ns = r[1]
            last_sync = time.monotonic()

        cv2.imshow("NDI", cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))
        if cv2.waitKey(1) & 0xFF == 27:           # ESC to quit
           break
except KeyboardInterrupt:
    pass
finally:
    ndi.recv_destroy(receiver)
    ndi.destroy()
    cv2.destroyAllWindows()