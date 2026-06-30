import cv2
import numpy as np
import time
import platform
from pathlib import Path
from ultralytics import YOLO
from utils import draw_detection, draw_fps, get_device

MODEL_PATH = "models/yolov8s.pt"
CONF_THRESH = 0.25
IOU_THRESH = 0.45
IMG_SIZE = 640
SKIP_FRAMES = 2

def process_frame(model, frame, names: dict, device: str = "cpu") -> tuple:
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
            from utils import get_color
            color  = get_color(cls_id)
            draw_detection(frame, box.xyxy[0].tolist(), label, conf, color)
            count += 1
    return frame, count

def run_yolo_only(cam_index: int = 0, width: int = 640, height: int = 480) -> None:
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

    print(f"[INFO] Kamera: {actual_w}x{actual_h} | Mode: YOLO Only")

    model = YOLO(MODEL_PATH)
    device = get_device()
    names = model.names
    print(f"[INFO] Model: {MODEL_PATH} | Device: {device.upper()}")

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

        if frame_count % SKIP_FRAMES == 0:
            frame, count = process_frame(model, frame, names, device)
            last_frame = frame.copy()
        elif last_frame is not None:
            frame = cv2.resize(last_frame, (actual_w, actual_h))
            count = 0
        else:
            count = 0

        elapsed = time.perf_counter() - t_start
        frame_times.append(elapsed)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_fps = len(frame_times) / max(sum(frame_times), 1e-6)

        draw_fps(frame, avg_fps)

        elapsed = time.perf_counter() - t_start
        sleep_time = max(0, FRAME_PERIOD - elapsed)
        time.sleep(sleep_time)
        frame_count += 1

        cv2.imshow("YOLO Detection | Q=Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Selesai — {frame_count} frame diproses")

if __name__ == "__main__":
    run_yolo_only()