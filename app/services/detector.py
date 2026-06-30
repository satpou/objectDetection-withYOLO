import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from utils import get_color, draw_detection

MODEL_PATH = "models/yolov8s.pt"
CONF_THRESH = 0.25
IOU_THRESH = 0.45
IMG_SIZE = 640

model = None
device = None

def init_model():
    global model, device
    if model is None:
        from utils import get_device
        model = YOLO(MODEL_PATH)
        device = get_device()
        print(f"[INFO] YOLO loaded: {device.upper()}")
    return model, device

def process_frame(frame):
    model, device = init_model()
    results = model.predict(
        source=frame, conf=CONF_THRESH, iou=IOU_THRESH,
        imgsz=IMG_SIZE, device=device, verbose=False,
    )[0]
    count = 0
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names.get(cls_id, str(cls_id))
            color = get_color(cls_id)
            draw_detection(frame, box.xyxy[0].tolist(), label, conf, color)
            count += 1
    return frame, count

def process_image(file_bytes):
    model, device = init_model()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, 0
    img, count = process_frame(img)
    return img, count

def process_video(input_path, output_path):
    model, device = init_model()
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    frame_count = 0
    total_objs = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, count = process_frame(frame)
        total_objs += count
        frame_count += 1
        text = f"Frame: {frame_count} | Objects: {total_objs}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        writer.write(frame)
    
    cap.release()
    writer.release()
    return frame_count, total_objs