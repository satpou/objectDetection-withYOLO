from pathlib import Path
from utils import draw_hand_landmarks

HAND_MODEL = "models/hand_landmarker.task"

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

landmarker = None

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
        print("[INFO] MediaPipe Hand loaded")
    return landmarker

def finger_up(tip, pip, landmarks):
    return landmarks[tip].y < landmarks[pip].y

def is_peace_gesture(lm):
    return (finger_up(8, 6, lm) and
            finger_up(12, 10, lm) and
            not finger_up(16, 14, lm) and
            not finger_up(20, 18, lm))

def is_open_palm(lm):
    return all(finger_up(t, t-2, lm) for t in [8, 12, 16, 20])

def detect(frame):
    landmarker = init_landmarker()
    if landmarker is None:
        return None, "MEDIAPIPE OFF"
    
    import cv2
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    
    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        if is_peace_gesture(lm):
            return lm, "PEACE"
        elif is_open_palm(lm):
            return lm, "OPEN PALM"
        else:
            return lm, "HAND"
    return None, "NO HAND"