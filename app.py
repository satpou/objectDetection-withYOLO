from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
from utils import draw_detection, draw_fps, draw_hand_landmarks, get_color, get_device

app = Flask(__name__)

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

model = None
landmarker = None
device = None
current_mode = "both"
camera = None

def init_model():
    global model, device
    if model is None:
        model = YOLO(MODEL_PATH)
        device = get_device()
        print(f"[INFO] Model loaded: {device.upper()}")

def init_landmarker():
    global landmarker
    if landmarker is None and MEDIAPIPE_AVAILABLE and Path(HAND_MODEL).exists():
        base_options = mp_tasks.BaseOptions(model_asset_path=HAND_MODEL)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        landmarker = mp_vision.HandLandmarker.create_from_options(options)
        print("[INFO] Landmarker loaded")

def finger_up(tip, pip, landmarks):
    return landmarks[tip].y < landmarks[pip].y

def is_peace_gesture(lm):
    return (finger_up(8, 6, lm) and
            finger_up(12, 10, lm) and
            not finger_up(16, 14, lm) and
            not finger_up(20, 18, lm))

def is_open_palm(lm):
    return all(finger_up(t, t-2, lm) for t in [8, 12, 16, 20])

def process_frame_yolo(frame):
    results = model.predict(
        source=frame, conf=CONF_THRESH, iou=IOU_THRESH,
        imgsz=IMG_SIZE, device=device, verbose=False,
    )[0]
    count = 0
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = model.names.get(cls_id, str(cls_id))
            color  = get_color(cls_id)
            draw_detection(frame, box.xyxy[0].tolist(), label, conf, color)
            count += 1
    return frame, count

def generate_frames():
    global camera, current_mode
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    blur_active = False
    last_frame = None
    frame_count = 0
    frame_times = []
    TARGET_FPS = 30
    
    while True:
        t_start = time.perf_counter()
        success, frame = camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        gesture_text = "NO HAND"
        
        if current_mode in ["hand", "both"] and landmarker is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            
            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                draw_hand_landmarks(frame, lm, w, h)
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
        
        if current_mode in ["yolo", "both"] and model is not None:
            if frame_count % SKIP_FRAMES == 0:
                frame, _ = process_frame_yolo(frame)
                last_frame = frame.copy()
            elif last_frame is not None:
                frame = cv2.resize(last_frame, (w, h))
        
        if blur_active:
            frame = cv2.GaussianBlur(frame, (61, 61), 0)
            cv2.putText(frame, "BLUR ON", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 2, cv2.LINE_AA)
        
        if current_mode in ["hand", "both"] and landmarker:
            dbg = f"Hand: {gesture_text}"
            cv2.putText(frame, dbg, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 1, cv2.LINE_AA)
        
        elapsed = time.perf_counter() - t_start
        frame_times.append(elapsed)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_fps = len(frame_times) / max(sum(frame_times), 1e-6)
        draw_fps(frame, avg_fps)
        
        frame_count += 1
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global current_mode
    data = request.get_json()
    mode = data.get('mode', 'both')
    
    if mode in ['yolo', 'hand', 'both']:
        current_mode = mode
        
        if mode in ['yolo', 'both']:
            init_model()
        if mode in ['hand', 'both']:
            init_landmarker()
        
        return jsonify({'status': 'success', 'mode': current_mode})
    return jsonify({'status': 'error', 'message': 'Invalid mode'})

@app.route('/get_status')
def get_status():
    return jsonify({
        'mode': current_mode,
        'mediapipe_available': MEDIAPIPE_AVAILABLE,
        'model_loaded': model is not None,
        'landmarker_loaded': landmarker is not None
    })

if __name__ == '__main__':
    init_model()
    init_landmarker()
    print("[INFO] Starting web server at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)