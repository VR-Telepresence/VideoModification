import asyncio
import cv2
import json
import websockets
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

# 360°-Panorama aktivieren
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2880)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
cap.set(cv2.CAP_PROP_FPS, 30)

async def handler(websocket):
    print("Unity connected!")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, classes=[0], verbose=False)[0]
            person_polygons = []

            if results.masks is not None and len(results.masks) > 0:
                for mask in results.masks:
                    pts = mask.xyn[0].tolist()
                    if len(pts) >= 3:
                        person_polygons.append({
                            "label": "person",
                            "confidence": 1.0,
                            "polygon": [{"x": float(x), "y": float(y)} for x, y in pts]
                        })

            # 4. JSON-Paket an Unity senden
            payload = {
                "type": "segmentation",
                "objects": person_polygons
            }
            await websocket.send(json.dumps(payload))

            _, jpeg_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            await websocket.send(jpeg_bytes.tobytes())

            #await asyncio.sleep(0.033) # ~30 FPS

    except websockets.exceptions.ConnectionClosed:
        print("Unity disconnected!")

async def main():
    async with websockets.serve(handler, "127.0.0.1", 8766):
        print("Segmentation server runs on ws://127.0.0.1:8765 ...")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())