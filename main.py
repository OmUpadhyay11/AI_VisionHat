# main.py
import time
import math
import cv2
from ultralytics import YOLO

from speech import speaker

# =========================
# SPEED / DEMO SETTINGS
# =========================
IMG_SIZE = 320          # try 256 if still slow
CONF = 0.55             # try 0.60 if too many false detections
FRAME_SKIP = 1          # 1=every frame, 2=every 2nd frame
SPEAK_COOLDOWN = 1.1    # seconds between spoken alerts

# =========================
# WHAT TO DETECT / PRIORITY
# =========================
# Only these classes will be considered for "named" callouts.
CLASS_WEIGHT = {
    "person": 6.0,
    "chair": 2.5,
    "couch": 2.5,
    "bench": 2.2,
    "tv": 1.8,
    "bottle": 1.5,
    "backpack": 1.6,
    "teddy bear": 1.0,
}

# bbox area ratio thresholds: (box_area / frame_area)
CLOSE_THR = 0.12
APPROACH_THR = 0.05

# =========================
# GENERIC "WALL/OBSTACLE" FALLBACK
# =========================
# If something fills a huge chunk of the view, we warn even if YOLO doesn't have a "wall" class.
OBSTACLE_AREA_THR = 0.35      # >= 35% of the frame is basically "something big in front"
OBSTACLE_CENTER_MIN = 0.55    # must be fairly centered (in your path)
OBSTACLE_WEIGHT = 4.0         # below person (6.0) but above chairs etc.

# =========================
# PHRASES
# =========================
def phrase(label: str, state: str) -> str:
    if label == "person":
        if state == "close":
            return "Human close. Please turn."
        if state == "approaching":
            return "Human approaching."
        return "Human far."

    if label == "obstacle":
        if state == "close":
            return "Obstacle very close. Please stop."
        if state == "approaching":
            return "Obstacle ahead."
        return "Obstacle far."

    if state == "close":
        return f"{label} close."
    if state == "approaching":
        return f"{label} approaching."
    return f"{label} far."

# =========================
# SCORING / PRIORITY
# =========================
def state_from_area_ratio(a: float) -> str:
    if a >= CLOSE_THR:
        return "close"
    if a >= APPROACH_THR:
        return "approaching"
    return "far"

def center_bonus(xc_norm: float, yc_norm: float) -> float:
    # 0.2..1.0 (higher if near center)
    dx = abs(xc_norm - 0.5)
    dy = abs(yc_norm - 0.5)
    d = math.sqrt(dx * dx + dy * dy)
    return max(0.2, 1.0 - d * 1.35)

def pick_top_object(result, frame_w: int, frame_h: int, names: dict):
    """
    Returns best item:
      (score, label, state, area_ratio)
    label can be one of CLASS_WEIGHT keys OR "obstacle"
    """
    frame_area = float(frame_w * frame_h)
    best = None

    if result.boxes is None:
        return None

    # 1) Normal class-based scoring (person/chair/etc.)
    for b in result.boxes:
        cls_id = int(b.cls[0])
        label = names.get(cls_id, str(cls_id))

        x1, y1, x2, y2 = b.xyxy[0].tolist()
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        area_ratio = (bw * bh) / frame_area
        xc = ((x1 + x2) * 0.5) / frame_w
        yc = ((y1 + y2) * 0.5) / frame_h
        cbonus = center_bonus(xc, yc)

        # Optional: reduce false "person" calls (tiny people are often wrong)
        if label == "person" and area_ratio < 0.03:
            continue

        if label in CLASS_WEIGHT:
            st = state_from_area_ratio(area_ratio)
            closeness = min(1.0, area_ratio / CLOSE_THR)
            score = CLASS_WEIGHT[label] * (0.55 + 0.45 * closeness) * cbonus

            if best is None or score > best[0]:
                best = (score, label, st, area_ratio)

    # 2) Generic obstacle fallback (helps for walls / large surfaces)
    # If any detected box is HUGE and centered, warn even if YOLO calls it something random.
    for b in result.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        area_ratio = (bw * bh) / frame_area

        xc = ((x1 + x2) * 0.5) / frame_w
        yc = ((y1 + y2) * 0.5) / frame_h
        cbonus = center_bonus(xc, yc)

        if area_ratio >= OBSTACLE_AREA_THR and cbonus >= OBSTACLE_CENTER_MIN:
            # treat as obstacle (state based on close/approach thresholds)
            st = state_from_area_ratio(area_ratio)
            closeness = min(1.0, area_ratio / CLOSE_THR)
            score = OBSTACLE_WEIGHT * (0.55 + 0.45 * closeness) * cbonus

            # only override if it's actually more important than current best
            if best is None or score > best[0]:
                best = (score, "obstacle", st, area_ratio)

    return best

def main():
    print("[VisionHat] starting...")

    model = YOLO("yolov8n.pt")
    names = model.names

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[VisionHat] ERROR: camera not opened")
        return

    # Speed win: lower resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)

    last_spoken_key = None
    last_spoken_time = 0.0

    frame_i = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[VisionHat] WARNING: frame read failed")
                time.sleep(0.05)
                continue

            frame_i += 1
            if FRAME_SKIP > 1 and (frame_i % FRAME_SKIP) != 0:
                continue

            h, w = frame.shape[:2]

            results = model.predict(
                source=frame,
                imgsz=IMG_SIZE,
                conf=CONF,
                verbose=False
            )

            r0 = results[0]
            top = pick_top_object(r0, w, h, names)

            now = time.time()
            if top is not None:
                _, label, st, _ = top
                key = (label, st)
                urgent = (st == "close")

                if (now - last_spoken_time) >= SPEAK_COOLDOWN and (key != last_spoken_key or urgent):
                    speaker.speak(phrase(label, st), cooldown=SPEAK_COOLDOWN, force=urgent)
                    last_spoken_time = now
                    last_spoken_key = key

    except KeyboardInterrupt:
        print("\n[VisionHat] stopping (Ctrl+C)")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()