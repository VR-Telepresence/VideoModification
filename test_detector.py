import sys, time, cv2
from detection import create_detector

path  = sys.argv[1] if len(sys.argv) > 1 else "vase.jpg"
model = sys.argv[2] if len(sys.argv) > 2 else "yolo"

img = cv2.resize(cv2.imread(path), (512, 512))
det = create_detector(model, every_n=1 if model == "yolo" else 5, style="outline")

t0, n = time.time(), 0
while True:
    det.submit(img)
    cv2.imshow(f"test: {model} (q = beenden)", cv2.resize(det.draw(img.copy()), (768, 768)))
    n += 1
    if n % 30 == 0:
        print(f"{n / (time.time() - t0):.1f} Loop-FPS")
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()