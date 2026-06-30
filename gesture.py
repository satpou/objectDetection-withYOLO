import cv2
import numpy as np
import time
from pathlib import Path

HAND_MODEL = "models/hand_landmarker.task"

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

def finger_up(tip, pip, landmarks) -> bool:
    return landmarks[tip].y < landmarks[pip].y

def is_peace_gesture(lm) -> bool:
    return (finger_up(8, 6, lm) and
            finger_up(12, 10, lm) and
            not finger_up(16, 14, lm) and
            not finger_up(20, 18, lm))

def is_open_palm(lm) -> bool:
    return all(finger_up(t, t-2, lm) for t in [8, 12, 16, 20])

def run_gesture_only(cam_index: int = 0, width: int = 640, height: int = 480) -> None:
    import platform
    from utils import draw_hand_landmarks

    system = platform.system()
    if system == "Darwin":
        cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
    elif system == "Windows":
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera index {cam_index} tidak dapat dibuka.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if actual_w == 0 or actual_h == 0:
        ret, test_frame = cap.read()
        if ret:
            actual_h, actual_w = test_frame.shape[:2]

    print(f"[INFO] Kamera: {actual_w}x{actual_h} | Mode: Hand Gesture Only")

    landmarker = None
    if MEDIAPIPE_AVAILABLE and Path(HAND_MODEL).exists():
        base_options = mp_tasks.BaseOptions(model_asset_path=HAND_MODEL)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=1,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5
        )
        landmarker = mp_vision.HandLandmarker.create_from_options(options)
        print("[INFO] Peace = blur ON, Open palm = blur OFF")
    else:
        print("[WARN] MediaPipe tidak tersedia")
        return

    blur_active = False
    frame_count = 0
    TARGET_FPS = 30
    FRAME_PERIOD = 1.0 / TARGET_FPS

    while True:
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        gesture_text = "NO HAND"
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            draw_hand_landmarks(frame, lm, actual_w, actual_h)
            if is_peace_gesture(lm):
                blur_active = True
                gesture_text = "PEACE"
            elif is_open_palm(lm):
                blur_active = False
                gesture_text = "OPEN PALM"
            else:
                blur_active = False
                gesture_text = "HAND"
        else:
            blur_active = False

        if blur_active:
            frame = cv2.GaussianBlur(frame, (61, 61), 0)
            cv2.putText(frame, "BLUR ON", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 2, cv2.LINE_AA)

        dbg = f"Hand: {'YES' if result.hand_landmarks else 'NO'} | {gesture_text}"
        cv2.putText(frame, dbg, (10, actual_h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1, cv2.LINE_AA)

        elapsed = time.perf_counter() - t_start
        sleep_time = max(0, FRAME_PERIOD - elapsed)
        time.sleep(sleep_time)
        frame_count += 1

        cv2.imshow("Hand Gesture Blur | Q=Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print(f"[INFO] Selesai — {frame_count} frame diproses")

if __name__ == "__main__":
    run_gesture_only()