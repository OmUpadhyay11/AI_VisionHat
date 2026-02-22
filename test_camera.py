import time
import cv2
from ultralytics import YOLO

# Load model (downloads yolov8n.pt first time)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Camera failed to open (/dev/video0).")

print("YOLO webcam test running (headless). Ctrl+C to stop.")

t0 = time.time()
frames = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frames += 1

        # Inference
        results = model.predict(frame, imgsz=640, conf=0.35, verbose=False)
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            # Print top 5 detections
            out = []
            for b in boxes[:5]:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                out.append(f"{model.names[cls]}({conf:.2f})")
            print("[det]", ", ".join(out))
        else:
            print("[det] none")

        # FPS every ~2 seconds
        if time.time() - t0 >= 2.0:
            fps = frames / (time.time() - t0)
            print(f"[fps] {fps:.1f}")
            t0 = time.time()
            frames = 0

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    cap.release()