# 🎯 Real-Time Object Detection with YOLOv8

Aplikasi **Real-Time Object Detection** menggunakan **Python**, **OpenCV**, dan **Ultralytics YOLOv8**. Project ini dapat melakukan deteksi objek melalui **webcam**, **gambar**, maupun **video** dengan performa yang ringan dan mudah digunakan.

---

## ✨ Features

- 📷 Real-time object detection dari webcam
- 🖼️ Deteksi objek pada gambar
- 🎥 Deteksi objek pada video
- ⚡ Frame skipping untuk meningkatkan performa
- 📊 Menampilkan FPS secara real-time
- 💾 Save hasil deteksi ke image atau video
- 🖥️ Support Windows, macOS, dan Linux

---

## 🛠️ Tech Stack

- Python 3.10+
- OpenCV
- Ultralytics YOLOv8
- NumPy
- PyTorch

---

## 📁 Project Structure

```text
.
├── models/
│   └── yolov8s.pt
├── assets/
│   ├── images/
│   └── videos/
├── output/
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone repository terlebih dahulu.

```bash
git clone https://github.com/satpou/yolo-object-detection.git
cd yolo-object-detection
```

(Optional) Buat virtual environment.

```bash
python -m venv .venv
```

Aktifkan virtual environment.

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install seluruh dependency.

```bash
pip install -r requirements.txt
```

Download model YOLOv8 dan simpan ke folder `models/`.

```
models/
└── yolov8s.pt
```

---

## 🚀 Cara Menjalankan

### Webcam Detection

```bash
python main.py webcam
```

Menyimpan hasil webcam.

```bash
python main.py webcam --save
```

---

### Image Detection

```bash
python main.py image assets/images/sample.jpg
```

Menentukan lokasi output.

```bash
python main.py image assets/images/sample.jpg --output output/result.jpg
```

---

### Video Detection

```bash
python main.py video assets/videos/sample.mp4
```

Menyimpan hasil ke lokasi tertentu.

```bash
python main.py video assets/videos/sample.mp4 --output output/result.mp4
```

---

## ⚙️ Konfigurasi

Beberapa konfigurasi dapat diubah langsung pada source code.

| Parameter | Keterangan |
|-----------|------------|
| `MODEL_PATH` | Lokasi model YOLO |
| `CONF_THRESH` | Minimum confidence |
| `IOU_THRESH` | IoU threshold |
| `IMG_SIZE` | Ukuran input image |
| `SKIP_FRAMES` | Jumlah frame yang dilewati |

---

## 📸 Demo

🚧 Coming Soon...

---

## 📌 Roadmap

- [ ] Object Counting
- [ ] Object Tracking (ByteTrack)
- [ ] Custom YOLO Model
- [ ] GUI Desktop
- [ ] Benchmark FPS
- [ ] Export Detection Log
- [ ] Docker Support

---

## 🤝 Contributing

Pull Request maupun Issue sangat terbuka apabila ingin memberikan saran atau pengembangan project ini.

---

## 📄 License

Project ini menggunakan **MIT License**.

---

## 👨‍💻 Author

**Satria Rahmaddhani**

GitHub: https://github.com/satpou