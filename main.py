"""
Real-Time Object Detection with YOLO
Author  : Satpou
Version : 1.0.0
"""

import cv2
import numpy as np
import argparse
import time
import platform
from pathlib import Path
from typing import Optional
from ultralytics import YOLO

# ── Config ────────────────────────────────────
MODEL_PATH  = "models/yolov8s.pt"
CONF_THRESH = 0.25   # minimum confidence
IOU_THRESH  = 0.45
IMG_SIZE    = 640
SKIP_FRAMES = 2      # inferensi setiap N frame
FONT        = cv2.FONT_HERSHEY_SIMPLEX

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


def get_device() -> str:
    # Pilih device terbaik: MPS (Mac) → CUDA (NVIDIA) → CPU
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def draw_detection(frame, box, label: str, conf: float, color: tuple) -> None:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label}  {conf:.0%}"
    (tw, th), baseline = cv2.getTextSize(text, FONT, 0.55, 1)
    bg_y1 = max(y1 - th - baseline - 6, 0)
    bg_y2 = max(y1, th + baseline + 6)
    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 8, bg_y2), color, cv2.FILLED)
    cv2.putText(frame, text, (x1 + 4, bg_y2 - baseline - 2),
                FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # confidence bar
    bar_h  = 4
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


def process_frame(model, frame, names: dict, device: str = "cpu") -> "tuple[np.ndarray, int]":
    results = model.predict(
        source=frame,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=IMG_SIZE,
        device=device,
        verbose=False,
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


def resolve_output_path(output_arg: Optional[str], src_stem: str, suffix: str, fallback_dir: Path) -> Path:
    default_name = f"{src_stem}_detected{suffix}"

    if output_arg is None:
        return fallback_dir / default_name

    out = Path(output_arg)
    if out.is_dir() or out.suffix == "":
        out.mkdir(parents=True, exist_ok=True)
        return out / default_name

    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def detect_webcam(
    model,
    cam_index: int = 0,
    width: int = 1280,
    height: int = 720,
    fps_smooth: int = 30,
    output: Optional[str] = None,
    save: bool = False,
) -> None:
    # Pilih backend kamera sesuai OS
    system = platform.system()
    if system == "Darwin":
        cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
    elif system == "Windows":
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)  # fallback
    if not cap.isOpened():
        raise RuntimeError(f"Kamera index {cam_index} tidak dapat dibuka.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if actual_w == 0 or actual_h == 0:
        ret, test_frame = cap.read()
        if ret:
            actual_h, actual_w = test_frame.shape[:2]
        else:
            raise RuntimeError("Kamera terbuka tapi tidak bisa membaca frame.")

    print(f"[INFO] Kamera       : {cam_index} ({actual_w}x{actual_h})")
    print(f"[INFO] OS           : {platform.system()} {platform.machine()}")

    writer   = None
    out_path = None
    if output is not None or save:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path  = resolve_output_path(output, f"webcam_{timestamp}", ".mp4", Path.cwd())
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
        cam_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer    = cv2.VideoWriter(str(out_path), fourcc, cam_fps, (actual_w, actual_h))
        print(f"[INFO] Merekam ke   : {out_path}")

    device = get_device()
    print(f"[INFO] Device       : {device.upper()}")
    print("[INFO] Tekan Q untuk keluar")

    names        = model.names
    frame_times  = []
    total_frames = 0
    total_objs   = 0
    last_frame   = None

    while True:
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame tidak terbaca, mencoba lagi...")
            for _ in range(5):
                ret, frame = cap.read()
                if ret:
                    break
            if not ret:
                print("[ERROR] Kamera berhenti mengirim frame.")
                break

        # Frame skipping — hemat komputasi
        if total_frames % SKIP_FRAMES == 0:
            frame, count = process_frame(model, frame, names, device)
            last_frame   = frame.copy()
        else:
            frame = last_frame if last_frame is not None else frame
            count = 0

        elapsed = time.perf_counter() - t_start
        frame_times.append(elapsed)
        if len(frame_times) > fps_smooth:
            frame_times.pop(0)
        avg_fps = len(frame_times) / max(sum(frame_times), 1e-6)

        draw_fps(frame, avg_fps)
        cv2.imshow("YOLO — Webcam Detection  |  Q to quit", frame)

        if writer is not None:
            writer.write(frame)

        total_frames += 1
        total_objs   += count
        if total_frames % 30 == 0:
            print(f"\r[Frame {total_frames:>5}]  FPS: {avg_fps:5.1f}  Obj/frame: {total_objs/total_frames:.1f}", end="")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\n[INFO] Video disimpan ke: {out_path}")
    cv2.destroyAllWindows()
    print(f"\n[INFO] Selesai — {total_frames} frame, rata-rata {total_objs/max(total_frames,1):.1f} objek/frame")


def detect_video(model, video_path: str, output: Optional[str] = None) -> None:
    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")

    cap     = cv2.VideoCapture(str(src))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = resolve_output_path(output, src.stem, src.suffix or ".mp4", src.parent)
    writer   = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))
    names    = model.names
    frame_n  = 0
    t_start  = time.perf_counter()

    print(f"[INFO] Video  : {src.name} ({total} frame)")
    print(f"[INFO] Output : {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, _  = process_frame(model, frame, names)
        elapsed   = time.perf_counter() - t_start
        fps_live  = (frame_n + 1) / max(elapsed, 1e-6)
        draw_fps(frame, fps_live)

        writer.write(frame)
        cv2.imshow(f"YOLO — {src.name}  |  Q to quit", frame)

        frame_n += 1
        print(f"\r[{frame_n:>{len(str(total))}}/{total}]  {fps_live:.1f} fps", end="")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Video disimpan ke: {out_path}")


def detect_image(model, image_path: str, output: Optional[str] = None) -> None:
    src = Path(image_path)
    if not src.exists():
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    frame = cv2.imread(str(src))
    if frame is None:
        raise ValueError(f"Tidak dapat membaca gambar: {image_path}")

    names        = model.names
    t0           = time.perf_counter()
    frame, count = process_frame(model, frame, names)
    elapsed_ms   = (time.perf_counter() - t0) * 1000

    draw_fps(frame, 1000 / max(elapsed_ms, 1e-3))

    out_path = resolve_output_path(output, src.stem, src.suffix or ".jpg", src.parent)
    cv2.imwrite(str(out_path), frame)

    print(f"[INFO] Terdeteksi {count} objek dalam {elapsed_ms:.1f} ms")
    print(f"[INFO] Gambar disimpan ke: {out_path}")

    cv2.imshow(f"YOLO — {src.name}  |  sembarang tombol untuk tutup", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-Time Object Detection with YOLOv8")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_cam = subparsers.add_parser("webcam")
    p_cam.add_argument("--cam",    type=int, default=0)
    p_cam.add_argument("--model",  type=str, default=MODEL_PATH)
    p_cam.add_argument("--save",   action="store_true")
    p_cam.add_argument("--output", type=str, default=None)

    p_vid = subparsers.add_parser("video")
    p_vid.add_argument("input",            type=str)
    p_vid.add_argument("--model",  type=str, default=MODEL_PATH)
    p_vid.add_argument("--output", type=str, default=None)

    p_img = subparsers.add_parser("image")
    p_img.add_argument("input",            type=str)
    p_img.add_argument("--model",  type=str, default=MODEL_PATH)
    p_img.add_argument("--output", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args  = parse_args()
    model = YOLO(args.model)
    print(f"[INFO] Model  : {args.model}")

    if args.mode == "webcam":
        detect_webcam(model, args.cam, output=args.output, save=args.save)
    elif args.mode == "video":
        detect_video(model, args.input, output=args.output)
    elif args.mode == "image":
        detect_image(model, args.input, output=args.output)


if __name__ == "__main__":
    main()