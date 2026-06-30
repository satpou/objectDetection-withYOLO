# Project Structure for Railway Deployment

railway-object-detection/
├── app.py                 # Main Flask app (entry point)
├── config.py              # Configuration
├── requirements.txt       # Python dependencies
├── Procfile               # Railway process definition
├── railway.json           # Railway config
├── .dockerignore          # Docker ignore (optional)
├── .env.example           # Environment variables template
├── README.md              # Documentation
├── app/
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── webcam.py
│   │   ├── upload.py
│   │   └── status.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── camera.py
│   │   ├── detector.py
│   │   └── gesture.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── drawing.py
│   └── templates/
│       └── index.html
├── models/
│   ├── yolov8s.pt
│   └── hand_landmarker.task
├── static/
│   ├── style.css
│   ├── script.js
│   └── upload.js
└── static/uploads/        # Created at runtime