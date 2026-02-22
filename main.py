import cv2
import numpy as np
from ultralytics import YOLO

from proximity import ProximityLogic
from speech import Speaker

CONF_THRES = 0.67
CENTER_WEIGHT = 1.0
AREA_WEIGHT = 0.2


def pick_front_object(result, frame_w, frame_h):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    cx_frame, cy_frame = frame_w / 2, frame_h / 2
    best = None
    best_score = -1e9

    for b in boxes:
        conf = float(b.conf[0])
        if conf < CONF_THRES:
            continue

        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        dist = np.sqrt(((cx - cx_frame) / frame_w) ** 2 + ((cy - cy_frame) / frame_h) ** 2)

        rel_area = (bw * bh) / (frame_w * frame_h)
        score = (1.0 - dist) * CENTER_WEIGHT + rel_area * AREA_WEIGHT

        if score > best_score:
            best_score = score
            cls_id = int(b.cls[0])
            label = result.names[cls_id]
            best = {
                "label": label,
                "conf": conf,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "rel_area": float(rel_area),
                "dist_to_center": float(dist),
            }

    return best


def normalize_label(label: str) -> str:
    if label == "teddy bear":
        return "stuffed animal"
    if label in ("laptop", "keyboard", "tv"):
        return "computer"
    return label


def main():
    model = YOLO("yolov8n.pt")

    # Pi usually uses /dev/video0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera at index 0. Check /dev/video* and permissions.")

    # person=0, chair=56, dining table=60, tv=62, laptop=63, keyboard=66, teddy bear=77
    KEEP = [0, 56, 60, 62, 63, 66, 77]

    proximity = ProximityLogic(
        approaching_area=0.06,
        danger_area=0.14,
        approach_slope=0.002,
        cooldown_frames=20
    )

    speaker = Speaker(cooldown_sec=1.2)

    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            h, w = frame.shape[:2]

            # Only selected classes + only >= 0.67 confidence
            r0 = model(frame, verbose=False, classes=KEEP, conf=0.67)[0]

            front = pick_front_object(r0, w, h)
            annotated = r0.plot()

            if front:
                x1, y1, x2, y2 = front["bbox"]
                label = normalize_label(front["label"])
                rel_area = front["rel_area"]

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    annotated,
                    f"{label} area={rel_area:.3f} conf={front['conf']:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                if front["conf"] >= 0.67:
                    event = proximity.update(label, rel_area)
                    if event:
                        print(event.message)
                        speaker.say(event.message, force=(event.level == "danger"))

            # Headless debug: save a frame occasionally so you can inspect output
            frame_count += 1
            if frame_count % 30 == 0:
                cv2.imwrite("debug.jpg", annotated)

    finally:
        cap.release()
        speaker.close()


if __name__ == "__main__":
    main()