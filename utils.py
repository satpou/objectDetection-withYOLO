import cv2
import numpy as np
import time

FONT = cv2.FONT_HERSHEY_SIMPLEX

PALETTE = [
    (86, 180, 233),
    (230, 159,   0),
    ( 0, 158, 115),
    (240, 228,  66),
    (213,  94,   0),
    (204, 121, 167),
]

def get_color(class_id: int) -> tuple:
    return PALETTE[class_id % len(PALETTE)]

def draw_detection(frame, box, label: str, conf: float, color: tuple) -> None:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label}  {conf:.0%}"
    (tw, th), baseline = cv2.getTextSize(text, FONT, 0.55, 1)
    bg_y1 = max(y1 - th - baseline - 6, 0)
    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 8, y1), color, cv2.FILLED)
    cv2.putText(frame, text, (x1 + 4, y1 - baseline - 2),
                FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    bar_h = 4
    bar_x2 = x1 + int((x2 - x1) * conf)
    cv2.rectangle(frame, (x1, y2 - bar_h), (x2, y2), (50, 50, 50), cv2.FILLED)
    cv2.rectangle(frame, (x1, y2 - bar_h), (bar_x2, y2), color, cv2.FILLED)

def draw_fps(frame, fps: float) -> None:
    text = f"FPS : {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.65, 2)
    margin = 10
    x = frame.shape[1] - tw - margin
    y = th + margin
    cv2.rectangle(frame, (x - 4, margin - 2), (x + tw + 4, y + 4), (20, 20, 20), cv2.FILLED)
    cv2.putText(frame, text, (x, y), FONT, 0.65, (0, 255, 128), 2, cv2.LINE_AA)

def draw_hand_landmarks(frame, landmarks, w, h) -> None:
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
    ]
    for i, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    for a, b in HAND_CONNECTIONS:
        ax, ay = int(landmarks[a].x * w), int(landmarks[a].y * h)
        bx, by = int(landmarks[b].x * w), int(landmarks[b].y * h)
        cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)

def get_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"