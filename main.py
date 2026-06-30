"""
Real-Time Detection System
Author : SatPou
Version: 1.1.0
"""

from gesture import run_gesture_only
from detector import run_yolo_only
from pathlib import Path

def show_menu():
    print("\n" + "="*50)
    print("   REAL-TIME DETECTION SYSTEM")
    print("="*50)
    print("   1) YOLO Object Detection Only")
    print("   2) Hand Gesture Blur Only")
    print("   3) Both (YOLO + Gesture)")
    print("   Q) Quit")
    print("="*50)

def main():
    while True:
        show_menu()
        choice = input("Select mode [1/2/3/Q]: ").strip().lower()
        
        if choice == "q":
            print("Goodbye!")
            break
        elif choice == "1":
            print("\n[MENU] Starting YOLO Detection...")
            run_yolo_only()
        elif choice == "2":
            print("\n[MENU] Starting Hand Gesture Blur...")
            run_gesture_only()
        elif choice == "3":
            print("\n[MENU] Starting Both Modes...")
            run_both()
        else:
            print("Invalid choice. Try again.")

def run_both():
    import cv2
    import numpy as np
    import time
    import platform
    from ultralytics import YOLO
    from utils import draw_detection, draw_fps, draw_hand_landmarks, get_color, get_device

    HAND_MODEL = "models/hand_landmarker.task"
    MODEL_PATH = "models/yolov8s.pt"
    CONF_THRESH = 0.25
    IOU_THRESH = 0.45
    IMG_SIZE = 640
    SKIP_FRAMES = 2

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        MEDIAPIPE_AVAILABLE = True
    except Exception:
        MEDIAPIPE_AVAILABLE = False

    def finger_up(tip, pip, landmarks):
        return landmarks[tip].y < landmarks[pip].y

    def is_peace_gesture(lm):
        return (finger_up(8, 6, lm) and
                finger_up(12, 10, lm) and
                not finger_up(16, 14, lm) and
                not finger_up(20, 18, lm))

    def is_open_palm(lm):
        return all(finger_up(t, t-2, lm) for t in [8, 12, 16, 20])

    def process_frame(model, frame, names, device):
        results = model.predict(
            source=frame, conf=CONF_THRESH, iou=IOU_THRESH,
            imgsz=IMG_SIZE, device=device, verbose=False,
        )[0]
        count = 0
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = names.get(cls_id, str(cls_id))
                color  = get_color(cls_id)
                draw_detection(frame, box.xyxy[0].tolist(), label, conf, color)
                count += 1
        return frame, count

    system = platform.system()
    if system == "Darwin":
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    elif system == "Windows":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Kamera tidak dapat dibuka.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if actual_w == 0 or actual_h == 0:
        ret, test_frame = cap.read()
        if ret:
            actual_h, actual_w = test_frame.shape[:2]

    print(f"[INFO] Kamera: {actual_w}x{actual_h} | Mode: Both")

    model = YOLO(MODEL_PATH)
    device = get_device()
    names = model.names
    print(f"[INFO] Model: {MODEL_PATH} | Device: {device.upper()}")

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

    blur_active = False
    last_frame = None
    frame_count = 0
    TARGET_FPS = 30
    FRAME_PERIOD = 1.0 / TARGET_FPS
    frame_times = []

    while True:
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Hand detection (every frame for responsiveness)
        gesture_text = "NO HAND"
        if landmarker is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
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

        # YOLO detection (skip frames)
        if frame_count % SKIP_FRAMES == 0:
            frame, _ = process_frame(model, frame, names, device)
            last_frame = frame.copy()
        elif last_frame is not None:
            frame = cv2.resize(last_frame, (actual_w, actual_h))

        if blur_active:
            frame = cv2.GaussianBlur(frame, (61, 61), 0)
            cv2.putText(frame, "BLUR ON", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 2, cv2.LINE_AA)

        dbg = f"Hand: {'YES' if result.hand_landmarks else 'NO'} | {gesture_text}" if landmarker else "MediaPipe: OFF"
        cv2.putText(frame, dbg, (10, actual_h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1, cv2.LINE_AA)

        elapsed = time.perf_counter() - t_start
        frame_times.append(elapsed)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_fps = len(frame_times) / max(sum(frame_times), 1e-6)
        draw_fps(frame, avg_fps)

        sleep_time = max(0, FRAME_PERIOD - elapsed)
        time.sleep(sleep_time)
        frame_count += 1

        cv2.imshow("YOLO + Hand Gesture | Q=Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if landmarker:
        landmarker.close()
    cv2.destroyAllWindows()
    print(f"[INFO] Selesai — {frame_count} frame diproses")

if __name__ == "__main__":
    main()