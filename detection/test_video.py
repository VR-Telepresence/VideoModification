# test_video.py
import time, argparse, cv2
from detector import create_detector

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--model", default="yolo")
ap.add_argument("--only", default="person")
ap.add_argument("--maxw", type=int, default=960)
ap.add_argument("--sync", action="store_true")
ap.add_argument("--save", default=None)
args = ap.parse_args()

only = tuple(s.strip() for s in args.only.split(","))
if args.model == "yolo":
    det = create_detector("yolo", only=only, imgsz=640, every_n=1, style="outline")
else:
    det = create_detector("sam3", prompt=only[0], style="outline")

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise SystemExit(f"Video nicht lesbar: {args.video}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
writer = None
t0, n = time.time(), 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    if w > args.maxw:
        s = args.maxw / w
        frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    if args.sync:
        det.latest = det.infer(frame)
    else:
        det.submit(frame)
    out = det.draw(frame.copy())

    if args.save and writer is None:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (out.shape[1], out.shape[0]))
    if writer:
        writer.write(out)

    cv2.imshow("video test (q = beenden)", out)
    n += 1
    if cv2.waitKey(1 if args.sync else int(1000 / fps)) & 0xFF == ord("q"):
        break

print(f"{n / (time.time() - t0):.1f} FPS Verarbeitung")
cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()