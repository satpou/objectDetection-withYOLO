# 🎯 Real-Time Object Detection with YOLOv8

Aplikasi **Real-Time Object Detection** menggunakan Python, Flask, OpenCV, YOLOv8, dan MediaPipe. Deteksi objek real-time dari webcam, gambar, dan video, plus gesture hand blur.

**Deploy**: Railway.app | **UI**: Web (Flask + Bootstrap 5)

---

## ✨ Features

- 📷 Webcam real-time detection + hand gesture blur
- 🖼️ Upload image → YOLO detection
- 🎥 Upload video → frame-by-frame YOLO detection
- ✌️ **Peace gesture** → Gaussian blur ON
- ✋ **Open palm** → Gaussian blur OFF
- 📊 FPS counter, hand landmark overlay
- ⚡ Frame skipping untuk performa
- 🖥️ Responsive dark-theme UI (desktop + mobile)

---

## 🛠️ Tech Stack

- Python 3.9+
- Flask + Gunicorn
- OpenCV
- Ultralytics YOLOv8
- MediaPipe Hand Landmarker
- PyTorch
- Bootstrap 5
- Railway.app

---

## 📁 Project Structure

```
.
├── app/                     # Aplikasi package
│   ├── __init__.py          # Flask app factory
│   ├── routes/
│   │   ├── webcam.py        # Video feed (MJPEG)
│   │   ├── upload.py        # Image/video upload API
│   │   └── status.py        # Mode & status API
│   ├── services/
│   │   ├── camera.py        # Camera init & read
│   │   ├── detector.py      # YOLO detection
│   │   ├── gesture.py       # MediaPipe hand detection
│   │   └── stream.py        # MJPEG streaming generator
│   └── utils/
│       └── drawing.py       # OpenCV drawing helpers
├── models/
│   ├── yolov8s.pt           # YOLO weights
│   └── hand_landmarker.task # MediaPipe model
├── static/
│   ├── style.css
│   ├── script.js
│   └── upload.js
├── templates/
│   └── index.html
├── main.py                  # Entry point
├── Procfile                 # Railway process
├── railway.json             # Railway config
├── requirements.txt
└── readme.md
```

---

## 🚀 Installation

Clone:

```bash
git clone https://github.com/satpou/realTime-detectionSystem
cd realTime-detectionSystem
```

Virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

Dependencies:

```bash
pip install -r requirements.txt
```

Download model:

```bash
curl -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

---

## 🚀 Run

```bash
python main.py
```

Buka: `http://localhost:5001`

---

## ⚙️ Konfigurasi

| Parameter | File | Keterangan |
|-----------|------|------------|
| `MODEL_PATH` | `app/services/detector.py` | Lokasi model YOLO |
| `CONF_THRESH` | `app/services/detector.py` | Minimum confidence (0.25) |
| `IOU_THRESH` | `app/services/detector.py` | IoU threshold (0.45) |
| `IMG_SIZE` | `app/services/detector.py` | Ukuran input YOLO (640) |
| `SKIP_FRAMES` | `app/services/stream.py` | Frame skip untuk YOLO (2) |

---

## 🖐️ Gesture

| Gesture | Action |
|---------|--------|
| **Peace (✌️)** | Blur ON |
| **Open Palm (✋)** | Blur OFF |

---

## 🚢 Deploy (Railway)

1. Push ke GitHub
2. Buka [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Set `PORT=5001` di Environment Variables
4. Done.

---

## 📄 License

MIT License

---

## 👨‍💻 Author

**Satria Rahmaddhani**

GitHub: https://github.com/satpou