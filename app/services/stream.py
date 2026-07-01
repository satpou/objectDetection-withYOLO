import time
import cv2
from flask import Response
from app.services.camera import camera, read_frame
from app.services.detector import process_frame
from app.services.gesture import detect
from app.utils.drawing import draw_hand_landmarks, overlay_text

current_mode = "both"
blur_active = False
SKIP_FRAMES = 2

def generate_frames():
    global blur_active, current_mode
    last_frame = None
    frame_count = 0
    frame_times = []
    gesture_text = "NO HAND"

    while True:
        t_start = time.perf_counter()
        frame = read_frame()
        h, w = frame.shape[:2]

        # Gesture detection
        if current_mode in ["hand", "both"]:
            lm, gesture_text = detect(frame)
            hand_detected = lm is not None
            
            if hand_detected:
                draw_hand_landmarks(frame, lm, w, h)
                if gesture_text == "PEACE":
                    blur_active = True
                elif gesture_text == "OPEN PALM":
                    blur_active = False
                else:
                    blur_active = False
            else:
                blur_active = False
        else:
            blur_active = False
            gesture_text = "HAND MODE OFF"

        # YOLO detection
        if current_mode in ["yolo", "both"]:
            if frame_count % SKIP_FRAMES == 0:
                frame, _ = process_frame(frame)
                last_frame = frame.copy()
            elif last_frame is not None:
                frame = frame

        # Blur
        if blur_active:
            frame = cv2.GaussianBlur(frame, (61, 61), 0)

        # Overlay
        elapsed = time.perf_counter() - t_start
        frame_times.append(elapsed)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_fps = len(frame_times) / max(sum(frame_times), 1e-6)
        
        overlay_text(frame, blur_active, gesture_text, f"FPS: {avg_fps:.1f}")
        frame_count += 1

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')