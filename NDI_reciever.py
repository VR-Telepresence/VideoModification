import cv2
import numpy as np
import NDIlib as ndi

# pip install opencv-python numpy ndi-python  (and install the NDI runtime)

ndi.initialize()

# Find an NDI source on the network
finder = ndi.find_create_v2()
sources = []
while not sources:
    ndi.find_wait_for_sources(finder, 2000)
    sources = ndi.find_get_current_sources(finder)
print("Connecting to", sources[0].ndi_name)

# Connect and ask for BGRA frames
settings = ndi.RecvCreateV3()
settings.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
receiver = ndi.recv_create_v3(settings)
ndi.recv_connect(receiver, sources[0])
ndi.find_destroy(finder)

try:
    while True:
        frame_type, video, _audio, _meta = ndi.recv_capture_v2(receiver, 1000)
        if frame_type == ndi.FRAME_TYPE_VIDEO:
            frame = np.copy(video.data)               # copy before freeing NDI's buffer
            ndi.recv_free_video_v2(receiver, video)
            cv2.imshow("NDI", cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))
            if cv2.waitKey(1) & 0xFF == 27:           # ESC to quit
                break
except KeyboardInterrupt:
    pass
finally:
    ndi.recv_destroy(receiver)
    ndi.destroy()
    cv2.destroyAllWindows()